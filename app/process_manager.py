"""Cross-process model-version registry and worker process handles.

The machine may host **at most two model processes** at a time.  Every
published model version is served by its own OS process (a ``uvicorn`` server
started in a new process group).  This module owns the bookkeeping that
enforces the cap and the swap state machine.

Per version / slot::

    (vacant) --reserve--> loading --commit--> active --retire--> draining --exit--> idle
                               |
                               +--load/test failure--> failed (slot freed)

Rules encoded here:

* Two persistent slots (``slot-0``, ``slot-1``).  A slot is reusable only
  after its process has exited *and* every inference it accepted has actually
  finished.  Until then the slot is occupied, so a third load is refused.
* A slot holding a ``loading`` candidate and a slot holding the ``active``
  version are both occupied; only ``idle`` slots can be reserved.
* ``accept_work`` reads the active version and increments that slot's
  in-flight counter in one locked operation.  The returned version is the
  version the request must use, so an accepted request can never be moved to a
  different model by a swap that commits immediately afterwards.
* ``commit_swap`` atomically promotes the tested candidate and moves the old
  version to ``draining``.  The old process is stopped only by the drain
  coordinator after ``in_flight == 0``; only after the OS process exits is
  ``release_slot`` allowed.
* ``begin_swap`` claims a swap *keyed by the old active version*.  A second
  swap claiming the same old version while the first is still in progress is
  rejected before any process is started.  The API runs as a single process
  (enforced by a data-directory lock), so this one writer owns the budget.
* ``begin_quarantine`` is the emergency override for a *draining* version
  whose in-flight work is stuck.  It force-settles exactly the outstanding
  work items of that version (each work item is settled at most once, keyed
  by a per-request token and guarded by the slot generation) so their callers
  fail explicitly.  It does **not** free the slot: the slot stays occupied
  until the API has confirmed the OS process is gone.  A quarantined version
  is never the active version, so in-service inference is untouched.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

SLOT_COUNT = 2

STATE_LOADING = "loading"
STATE_ACTIVE = "active"
STATE_DRAINING = "draining"
STATE_IDLE = "idle"

VERSION_LOADING = "loading"
VERSION_ACTIVE = "active"
VERSION_RETIRED = "retired"
VERSION_FAILED = "failed"

# Quarantine lifecycle of a version.  ``none`` -> normal.  ``pending`` means
# the emergency settlement was recorded but the OS process has not been
# confirmed gone yet (the slot is still occupied).  ``released`` means the
# version was quarantined and its slot has since been freed.
QUARANTINE_NONE = "none"
QUARANTINE_PENDING = "pending"
QUARANTINE_RELEASED = "released"
QUARANTINE_FAILED = "failed"

CLAIM_NONE = "__none__"


class NoCapacity(RuntimeError):
    """Both process slots are occupied (loading/active/draining)."""


class SwapConflict(RuntimeError):
    """The swap's old version is no longer current, or a swap is in progress."""


class QuarantineConflict(RuntimeError):
    """Quarantine was requested for something that is not a draining version."""


class UnknownVersion(LookupError):
    """The version id is not in the registry."""


class NoActiveModel(RuntimeError):
    """Inference was requested before any model became active."""


class FeatureMismatch(ValueError):
    """The feature vector width does not match the active model."""

    def __init__(self, message: str, version: str, expected: int) -> None:
        super().__init__(message)
        self.version = version
        self.expected = expected


class Registry:
    """Manager-backed shared state; every operation is serialised by a lock."""

    def __init__(self, manager: Any) -> None:
        self._manager = manager
        self._lock = manager.RLock()
        self._versions = manager.dict()
        self._slots = manager.dict()
        self._claims = manager.dict()
        # One entry per accepted inference: {version, slot_id, generation,
        # settled, outcome, accepted_at}.  Settlement is keyed by this token so
        # a request can be settled exactly once even when the drain gate and an
        # emergency quarantine race, or when a late worker response arrives
        # after its version has been quarantined.
        self._work = manager.dict()
        self._state = manager.dict({"active_version": None})
        now = time.time()
        for i in range(SLOT_COUNT):
            slot_id = f"slot-{i}"
            self._slots[slot_id] = {
                "slot_id": slot_id,
                "version": None,
                "process_state": STATE_IDLE,
                "pid": None,
                "in_flight": 0,
                "accepted": 0,
                "completed": 0,
                "abandoned": 0,
                "started_at": None,
                "idle_since": now,
                "generation": 0,
                "quarantine_state": QUARANTINE_NONE,
            }

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            versions = {k: dict(v) for k, v in self._versions.items()}
            slots = {k: dict(v) for k, v in self._slots.items()}
            return {
                "active_version": self._state["active_version"],
                "capacity": SLOT_COUNT,
                "slots_in_use": sum(
                    1 for s in slots.values() if s["process_state"] != STATE_IDLE
                ),
                "slots": [slots[k] for k in sorted(slots)],
                "versions": versions,
                "claims": dict(self._claims),
            }

    def active_version(self) -> Optional[str]:
        with self._lock:
            return self._state["active_version"]

    def version_info(self, version: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            info = self._versions.get(version)
            return dict(info) if info is not None else None

    def version_in_flight(self, version: str) -> int:
        with self._lock:
            info = self._versions.get(version)
            if info is None:
                return 0
            slot = self._slots.get(info["slot_id"])
            if slot is None or slot["version"] != version:
                return 0
            return int(slot["in_flight"])

    def work_outcome(self, token: str) -> Optional[Dict[str, Any]]:
        """Settlement record for an accepted request, if it exists."""
        with self._lock:
            work = self._work.get(token)
            return dict(work) if work is not None else None

    def has_capacity(self) -> bool:
        with self._lock:
            return any(
                s["process_state"] == STATE_IDLE for s in self._slots.values()
            )

    # ------------------------------------------------------------------
    # Admission / registration
    # ------------------------------------------------------------------
    def begin_swap(
        self,
        expected_version: Optional[str],
        token: str,
        enforce_expected: bool = True,
    ) -> Optional[str]:
        """Claim the right to replace the current active version.

        ``expected_version`` is the version the caller observed as active when
        it decided to swap.  If ``enforce_expected`` is set and the active
        version has changed since then, the caller is stale and gets
        ``SwapConflict``.  The claim is keyed by the current old version, so of
        several swaps that observed the *same* old version at most one can
        proceed.
        """
        with self._lock:
            active = self._state["active_version"]
            if enforce_expected and expected_version != active:
                raise SwapConflict(
                    f"swap precondition failed: caller observed active version "
                    f"{expected_version!r}, but the current active version is {active!r}"
                )
            key = active if active is not None else CLAIM_NONE
            if key in self._claims:
                raise SwapConflict(
                    f"a swap from version {active!r} is already in progress"
                )
            self._claims[key] = token
            return active

    def end_swap(self, old_version: Optional[str], token: str) -> None:
        key = old_version if old_version is not None else CLAIM_NONE
        with self._lock:
            if self._claims.get(key) == token:
                del self._claims[key]

    def reserve_any_slot(self, version: str) -> str:
        """Atomically pick an idle slot and register a loading candidate.

        A version id whose previous attempt *failed* may be retried; the new
        attempt keeps the earlier error for auditing.  Any other existing
        version id is rejected, so a live or historical version can never be
        overwritten.
        """
        with self._lock:
            previous = self._versions.get(version)
            attempt = 1
            previous_error = None
            if previous is not None:
                if previous["state"] != VERSION_FAILED:
                    raise SwapConflict(
                        f"version {version!r} already exists ({previous['state']})"
                    )
                attempt = int(previous.get("attempt", 1)) + 1
                previous_error = previous.get("error")
            free = sorted(
                sid
                for sid, s in self._slots.items()
                if s["process_state"] == STATE_IDLE
            )
            if not free:
                raise NoCapacity(
                    "both model process slots are occupied (loading/active/"
                    "draining); wait for the draining version to finish"
                )
            slot_id = free[0]
            now = time.time()
            slot = dict(self._slots[slot_id])
            slot.update(
                version=version,
                process_state=STATE_LOADING,
                pid=None,
                in_flight=0,
                accepted=0,
                completed=0,
                abandoned=0,
                started_at=now,
                idle_since=None,
                generation=slot["generation"] + 1,
                quarantine_state=QUARANTINE_NONE,
            )
            self._slots[slot_id] = slot
            self._versions[version] = {
                "version": version,
                "slot_id": slot_id,
                "pid": None,
                "state": VERSION_LOADING,
                "process_state": STATE_LOADING,
                "created_at": now,
                "tested_at": None,
                "activated_at": None,
                "retired_at": None,
                "error": None,
                "attempt": attempt,
                "previous_error": previous_error,
                "quarantine_state": QUARANTINE_NONE,
                "quarantine_reason": None,
                "quarantined_at": None,
                "abandoned_work": 0,
            }
            return slot_id

    def set_pid(self, version: str, pid: int) -> None:
        with self._lock:
            info = self._versions.get(version)
            if info is None:
                return
            info = dict(info)
            info["pid"] = pid
            self._versions[version] = info
            slot = self._slots.get(info["slot_id"])
            if slot is not None and slot["version"] == version:
                slot = dict(slot)
                slot["pid"] = pid
                self._slots[info["slot_id"]] = slot

    def set_model_shape(self, version: str, n_features: int, n_classes: int) -> None:
        """Record the candidate's input/output dimensions for request validation."""
        with self._lock:
            info = self._versions.get(version)
            if info is None:
                return
            info = dict(info)
            info["n_features"] = int(n_features)
            info["n_classes"] = int(n_classes)
            self._versions[version] = info

    def mark_candidate_tested(self, version: str) -> None:
        with self._lock:
            info = self._versions.get(version)
            if info is None or info["state"] != VERSION_LOADING:
                raise SwapConflict(f"version {version!r} is not a loading candidate")
            info = dict(info)
            info["tested_at"] = time.time()
            self._versions[version] = info

    def fail_candidate(self, version: str, error: str) -> None:
        """Mark a loading candidate failed and free its slot (process exited)."""
        with self._lock:
            info = self._versions.get(version)
            if info is None or info["state"] != VERSION_LOADING:
                return
            info = dict(info)
            info.update(
                state=VERSION_FAILED,
                process_state=STATE_IDLE,
                error=error,
                retired_at=time.time(),
            )
            self._versions[version] = info
            slot = self._slots.get(info["slot_id"])
            if slot is not None and slot["version"] == version:
                slot = dict(slot)
                slot.update(
                    version=None,
                    process_state=STATE_IDLE,
                    pid=None,
                    in_flight=0,
                    idle_since=time.time(),
                )
                self._slots[info["slot_id"]] = slot

    # ------------------------------------------------------------------
    # Commit / inference / drain
    # ------------------------------------------------------------------
    def commit_swap(self, version: str, old_version: Optional[str], token: str) -> None:
        with self._lock:
            active = self._state["active_version"]
            if active != old_version:
                raise SwapConflict(
                    f"active version changed from {old_version!r} to {active!r}; "
                    "candidate was not promoted"
                )
            key = active if active is not None else CLAIM_NONE
            if self._claims.get(key) != token:
                raise SwapConflict("swap claim was lost")
            info = self._versions.get(version)
            if info is None or info["state"] != VERSION_LOADING:
                raise SwapConflict(f"version {version!r} is not a loading candidate")
            if not info.get("tested_at"):
                raise SwapConflict(f"version {version!r} has not passed its smoke test")

            if active is not None:
                old = dict(self._versions[active])
                old.update(
                    state=VERSION_RETIRED,
                    process_state=STATE_DRAINING,
                    retired_at=time.time(),
                )
                self._versions[active] = old
                old_slot = self._slots.get(old["slot_id"])
                if old_slot is not None and old_slot["version"] == active:
                    old_slot = dict(old_slot)
                    old_slot["process_state"] = STATE_DRAINING
                    self._slots[old["slot_id"]] = old_slot

            new = dict(info)
            new.update(
                state=VERSION_ACTIVE,
                process_state=STATE_ACTIVE,
                activated_at=time.time(),
            )
            self._versions[version] = new
            slot = self._slots.get(new["slot_id"])
            if slot is None or slot["version"] != version:
                raise SwapConflict("candidate slot ownership changed")
            slot = dict(slot)
            slot["process_state"] = STATE_ACTIVE
            self._slots[new["slot_id"]] = slot
            self._state["active_version"] = version
            del self._claims[key]

    def accept_work(
        self, n_features: Optional[int] = None, token: Optional[str] = None
    ) -> Tuple[str, str]:
        """Pin an inference to the active version and count it in-flight.

        With ``n_features`` the request is validated against the active
        version's input width inside the same critical section, so a rejected
        request never touches the in-flight counter.

        ``token`` is the settlement key.  It records the slot generation the
        request was admitted under, so a response that arrives after the slot
        has been reused can never settle the *new* process's counters.
        """
        with self._lock:
            version = self._state["active_version"]
            if version is None:
                raise NoActiveModel("no model is active")
            info = self._versions.get(version)
            if info is None or info["state"] != VERSION_ACTIVE:
                raise NoActiveModel(f"version {version!r} is not active")
            if (
                n_features is not None
                and info.get("n_features") is not None
                and int(info["n_features"]) != n_features
            ):
                raise FeatureMismatch(
                    f"version {version!r} expects {info['n_features']} features, "
                    f"got {n_features}",
                    version=version,
                    expected=int(info["n_features"]),
                )
            slot = self._slots.get(info["slot_id"])
            if (
                slot is None
                or slot["version"] != version
                or slot["process_state"] != STATE_ACTIVE
            ):
                raise NoActiveModel("active slot is inconsistent")
            slot = dict(slot)
            slot["in_flight"] += 1
            slot["accepted"] += 1
            self._slots[info["slot_id"]] = slot
            if token is not None:
                self._work[token] = {
                    "version": version,
                    "slot_id": info["slot_id"],
                    "generation": int(slot["generation"]),
                    "settled": False,
                    "outcome": None,
                    "accepted_at": time.time(),
                }
            return version, info["slot_id"]

    def complete_work(self, version: str, token: Optional[str] = None) -> bool:
        """Settle an accepted request exactly once.

        With ``token`` the settlement is conditional on the work item still
        being open *and* the slot still belonging to the same version **and
        generation**.  A late worker response for an already-quarantined
        request is therefore a no-op, and can never decrement the counters of a
        reused slot.  Returns True when this call performed the settlement.
        """
        with self._lock:
            if token is not None:
                work = self._work.get(token)
                if work is None:
                    return False
                if work["settled"]:
                    if work.get("outcome") == "quarantined":
                        self._note_late_completion(version, token, work)
                    return False
                info = self._versions.get(version)
                if info is None:
                    return False
                slot = self._slots.get(info["slot_id"])
                if (
                    slot is None
                    or slot["version"] != version
                    or int(slot["generation"]) != int(work["generation"])
                ):
                    # The slot was reused: nothing here may be mutated.
                    return False
                slot = dict(slot)
                slot["in_flight"] = max(0, slot["in_flight"] - 1)
                slot["completed"] += 1
                self._slots[info["slot_id"]] = slot
                work = dict(work)
                work.update(
                    settled=True,
                    outcome="completed",
                    settled_at=time.time(),
                    late_completion=False,
                )
                self._work[token] = work
                return True
            return self._complete_unkeyed(version)

    def _note_late_completion(
        self, version: str, token: str, work: Dict[str, Any]
    ) -> None:
        """Flag a worker response that returned after its request was abandoned.

        The caller was already failed by the quarantine, so the response must
        not settle anything a second time.  If the slot is still owned by the
        same version *and generation* the completion is recorded as pending so
        that ``abort_quarantine`` can apply it exactly once if the emergency
        stop fails; once the slot is released/reused it is a no-op.
        """
        info = self._versions.get(version)
        if info is None:
            return
        slot = self._slots.get(info["slot_id"])
        if (
            slot is None
            or slot["version"] != version
            or int(slot["generation"]) != int(work["generation"])
        ):
            return
        work = dict(work)
        work["late_completion"] = True
        work["late_completion_at"] = time.time()
        self._work[token] = work

    def _complete_unkeyed(self, version: str) -> bool:
        info = self._versions.get(version)
        if info is None:
            return False
        slot = self._slots.get(info["slot_id"])
        if slot is None or slot["version"] != version:
            return False
        slot = dict(slot)
        slot["in_flight"] = max(0, slot["in_flight"] - 1)
        slot["completed"] += 1
        self._slots[info["slot_id"]] = slot
        return True

    def begin_quarantine(self, version: str, reason: str) -> Dict[str, Any]:
        """Emergency isolation of a draining version.

        Only a ``retired`` version whose process is still ``draining`` may be
        quarantined: the active version keeps serving, and a normal drain is
        still allowed to wait for real completion.  All outstanding work items
        of that version are settled as ``"quarantined"`` in one critical
        section, so each caller fails explicitly and no request can be settled
        twice.

        The slot is **not** freed here and the in-flight count is **not**
        zeroed: if the emergency stop cannot be confirmed, the ordinary drain
        gate must keep waiting for the real completion.  ``release_slot`` only
        succeeds with ``process_exited=True``; callers that could not confirm
        exit use :meth:`abort_quarantine` instead.
        """
        with self._lock:
            info = self._versions.get(version)
            if info is None:
                raise UnknownVersion(f"unknown version {version!r}")
            if info["state"] != VERSION_RETIRED:
                raise QuarantineConflict(
                    f"version {version!r} is {info['state']!r}, not a retired "
                    "(draining) version; only a draining version can be quarantined"
                )
            slot = self._slots.get(info["slot_id"])
            if slot is None or slot["version"] != version:
                raise QuarantineConflict(
                    f"version {version!r} no longer owns its slot"
                )
            if slot["process_state"] != STATE_DRAINING:
                raise QuarantineConflict(
                    f"version {version!r} is not draining (slot is "
                    f"{slot['process_state']!r})"
                )
            already = info.get("quarantine_state") in (
                QUARANTINE_PENDING,
                QUARANTINE_RELEASED,
            )
            now = time.time()
            abandoned = 0
            abandoned_tokens: list[str] = []
            if not already:
                info = dict(info)
                info.update(
                    quarantine_state=QUARANTINE_PENDING,
                    quarantine_reason=reason,
                    quarantined_at=now,
                )
                self._versions[version] = info
                slot = dict(slot)
                slot["quarantine_state"] = QUARANTINE_PENDING
                self._slots[info["slot_id"]] = slot
                for token in list(self._work.keys()):
                    work = self._work.get(token)
                    if work is None or work["settled"] or work["version"] != version:
                        continue
                    if int(work["generation"]) != int(slot["generation"]):
                        continue
                    work = dict(work)
                    work.update(
                        settled=True,
                        outcome="quarantined",
                        settled_at=now,
                        late_completion=False,
                    )
                    self._work[token] = work
                    abandoned += 1
                    abandoned_tokens.append(token)
                if abandoned:
                    slot = dict(self._slots[info["slot_id"]])
                    slot["abandoned"] = int(slot.get("abandoned", 0)) + abandoned
                    self._slots[info["slot_id"]] = slot
                info = dict(self._versions.get(version))
                info["abandoned_work"] = int(info.get("abandoned_work", 0)) + abandoned
                self._versions[version] = info
            current_slot = self._slots[info["slot_id"]]
            return {
                "version": version,
                "slot_id": info["slot_id"],
                "quarantine_state": info["quarantine_state"],
                "reason": info.get("quarantine_reason"),
                "quarantined_at": info.get("quarantined_at"),
                "abandoned_work": int(info.get("abandoned_work", 0)),
                "in_flight": int(current_slot["in_flight"]),
                "already": already,
                "abandoned_tokens": abandoned_tokens,
            }

    def abort_quarantine(self, version: str, error: str) -> Dict[str, Any]:
        """Record that the emergency isolation could not be carried through.

        The slot is left occupied and the work accounting reverts to normal
        drain semantics: requests the caller already saw fail stay settled
        (and any response that arrived late is counted as completed exactly
        once), while work that is genuinely still running becomes pending
        again so the ordinary drain gate waits for it.
        """
        with self._lock:
            info = self._versions.get(version)
            if info is None:
                raise UnknownVersion(f"unknown version {version!r}")
            info = dict(info)
            info.update(
                quarantine_state=QUARANTINE_FAILED,
                error=error,
            )
            self._versions[version] = info
            slot = self._slots.get(info["slot_id"])
            if slot is not None and slot["version"] == version:
                slot = dict(slot)
                slot["quarantine_state"] = QUARANTINE_FAILED
                restored = 0
                reclaimed = 0
                for token in list(self._work.keys()):
                    work = self._work.get(token)
                    if (
                        work is None
                        or work["version"] != version
                        or work.get("outcome") != "quarantined"
                        or int(work["generation"]) != int(slot["generation"])
                    ):
                        continue
                    work = dict(work)
                    if work.get("late_completion"):
                        # The worker call already returned during the failed
                        # emergency window; settle it now, exactly once.
                        work.update(
                            settled=True,
                            outcome="completed",
                            settled_at=time.time(),
                        )
                        self._work[token] = work
                        slot["in_flight"] = max(0, slot["in_flight"] - 1)
                        slot["completed"] += 1
                        reclaimed += 1
                    else:
                        work.update(settled=False, outcome=None, late_completion=False)
                        self._work[token] = work
                        restored += 1
                # Work that was only provisionally abandoned and is still
                # genuinely running is no longer "abandoned": the normal drain
                # owns it again.
                slot["abandoned"] = max(0, int(slot.get("abandoned", 0)) - restored)
                self._slots[info["slot_id"]] = slot
                info["abandoned_work"] = max(
                    0, int(info.get("abandoned_work", 0)) - restored
                )
                self._versions[version] = info
            return {
                "version": version,
                "slot_id": info["slot_id"],
                "quarantine_state": QUARANTINE_FAILED,
                "error": error,
                "in_flight": int(self._slots[info["slot_id"]]["in_flight"]),
            }

    def mark_quarantine_released(self, version: str) -> None:
        with self._lock:
            info = self._versions.get(version)
            if info is None:
                return
            info = dict(info)
            info["quarantine_state"] = QUARANTINE_RELEASED
            self._versions[version] = info

    def release_slot(self, slot_id: str, version: str, process_exited: bool) -> bool:
        """Free a drained slot.

        ``process_exited`` must be True: slot reuse is gated on the OS process
        actually being gone, not on the API's intent to stop it.
        """
        if not process_exited:
            return False
        with self._lock:
            slot = self._slots.get(slot_id)
            if slot is None or slot["version"] != version:
                return False
            quarantined = slot.get("quarantine_state") == QUARANTINE_PENDING
            if slot["in_flight"] != 0:
                # Every counted request may be settled at most once.  For a
                # quarantined version the work was already explicitly settled,
                # so a non-zero counter is stale bookkeeping rather than real
                # work -- and the process is confirmed gone.  A normal drain,
                # by contrast, must keep waiting for the real completion.
                if not quarantined:
                    return False
            if slot["process_state"] not in (STATE_DRAINING, STATE_LOADING):
                return False
            slot = dict(slot)
            slot.update(
                version=None,
                process_state=STATE_IDLE,
                pid=None,
                in_flight=0,
                idle_since=time.time(),
            )
            # A quarantined version's marker must survive into the idle slot so
            # status queries can still report it.
            if slot.get("quarantine_state") == QUARANTINE_PENDING:
                slot["quarantine_state"] = QUARANTINE_RELEASED
            self._slots[slot_id] = slot
            info = self._versions.get(version)
            if info is not None:
                info = dict(info)
                info["process_state"] = STATE_IDLE
                if info.get("quarantine_state") == QUARANTINE_PENDING:
                    info["quarantine_state"] = QUARANTINE_RELEASED
                # A failed quarantine stays visible in status as "failed" after
                # the normal drain finally releases the slot.
                self._versions[version] = info
            return True

    def fail_active(self, version: str, error: str) -> bool:
        """Take a crashed active version out of service and free its slot.

        Used when the worker process dies unexpectedly.  Refuses while work is
        still counted in-flight (those requests will fail on their own and
        decrement the counter), and never touches a different active version.
        """
        with self._lock:
            info = self._versions.get(version)
            if info is None or info["state"] != VERSION_ACTIVE:
                return False
            slot = self._slots.get(info["slot_id"])
            if slot is None or slot["version"] != version or slot["in_flight"] != 0:
                return False
            info = dict(info)
            info.update(state=VERSION_FAILED, error=error, retired_at=time.time())
            self._versions[version] = info
            slot = dict(slot)
            slot.update(
                version=None,
                process_state=STATE_IDLE,
                pid=None,
                in_flight=0,
                idle_since=time.time(),
            )
            self._slots[info["slot_id"]] = slot
            if self._state["active_version"] == version:
                self._state["active_version"] = None
            return True

    def settle_dead_worker(self, version: str) -> bool:
        """If a worker died mid-request, stop counting its in-flight work.

        Called by the drain/reaper loop once the OS process is gone: the
        corresponding HTTP calls have failed or will fail, so keeping the
        counter raised would wedge the slot forever.
        """
        with self._lock:
            info = self._versions.get(version)
            if info is None:
                return False
            slot = self._slots.get(info["slot_id"])
            if slot is None or slot["version"] != version:
                return False
            if slot["in_flight"] == 0:
                return False
            slot = dict(slot)
            slot["completed"] += slot["in_flight"]
            slot["in_flight"] = 0
            self._slots[info["slot_id"]] = slot
            # Mark the outstanding work items settled so a late response can
            # never settle them a second time.
            for token in list(self._work.keys()):
                work = self._work.get(token)
                if work is None or work["settled"] or work["version"] != version:
                    continue
                if int(work["generation"]) != int(slot["generation"]):
                    continue
                work = dict(work)
                work.update(settled=True, outcome="worker_died", settled_at=time.time())
                self._work[token] = work
            return True


# ---------------------------------------------------------------------------
# API-side worker process handle
# ---------------------------------------------------------------------------


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


class WorkerHandle:
    """One model worker OS process and its private control directory."""

    def __init__(
        self,
        slot_id: str,
        version: str,
        model_path: str,
        work_dir: str,
        host: str = "127.0.0.1",
    ) -> None:
        self.slot_id = slot_id
        self.version = version
        self.model_path = model_path
        self.host = host
        self.port: Optional[int] = None
        self.pid: Optional[int] = None
        self._proc: Optional[subprocess.Popen] = None
        self.control_dir = os.path.join(work_dir, f"{slot_id}-{uuid.uuid4().hex[:8]}")
        os.makedirs(self.control_dir, exist_ok=True)
        self.ready_file = os.path.join(self.control_dir, "ready.json")

    def inference_url(self) -> str:
        if self.port is None:
            raise RuntimeError(f"worker {self.version} is not listening yet")
        return f"http://{self.host}:{self.port}/infer"

    def status_url(self) -> str:
        if self.port is None:
            raise RuntimeError(f"worker {self.version} is not listening yet")
        return f"http://{self.host}:{self.port}/status"

    def spawn(
        self,
        slow_load_ms: int = 0,
        simulate_load_failure: bool = False,
        load_timeout: float = 30.0,
    ) -> None:
        """Start the worker process (does not wait for readiness)."""
        root = str(Path(__file__).resolve().parent.parent)
        cmd = [
            sys.executable,
            "-m",
            "app.worker_launcher",
            self.slot_id,
            self.version,
            self.model_path,
            self.control_dir,
            self.ready_file,
            self.host,
            str(slow_load_ms),
            "1" if simulate_load_failure else "0",
        ]
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        self._proc = subprocess.Popen(
            cmd,
            start_new_session=True,
            cwd=root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        )
        self.pid = self._proc.pid

    def wait_ready(self, timeout: float = 30.0) -> Dict[str, Any]:
        """Block until the worker writes its ready handshake or exits."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            ready = _read_json(self.ready_file)
            if ready is not None:
                if ready.get("ok") and ready.get("port"):
                    self.port = int(ready["port"])
                return ready
            if self._proc is not None and self._proc.poll() is not None:
                raise RuntimeError(
                    f"worker exited during load (exit code {self._proc.returncode})"
                )
            time.sleep(0.02)
        raise TimeoutError(f"worker did not become ready within {timeout:.1f}s")

    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def wait_exit(self, timeout: float) -> bool:
        if self._proc is None:
            return True
        try:
            self._proc.wait(timeout=timeout)
            return True
        except subprocess.TimeoutExpired:
            return False

    def stop(self, timeout: float = 10.0) -> None:
        """Terminate the process group and reap the child."""
        if self._proc is None:
            return
        if self._proc.poll() is not None:
            self._proc.wait()
            return
        self._signal_group(signal.SIGTERM)
        if not self.wait_exit(timeout):
            self._signal_group(signal.SIGKILL)
            self.wait_exit(timeout)

    def _signal_group(self, sig: int) -> None:
        try:
            os.killpg(os.getpgid(self.pid), sig)  # type: ignore[arg-type]
        except (ProcessLookupError, PermissionError, TypeError):
            pass
