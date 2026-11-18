"""Unit tests for the shared registry state machine, including the drain gate."""

from __future__ import annotations

import multiprocessing

import pytest

from app.process_manager import (
    QUARANTINE_FAILED,
    QUARANTINE_NONE,
    QUARANTINE_PENDING,
    QUARANTINE_RELEASED,
    VERSION_ACTIVE,
    VERSION_FAILED,
    VERSION_RETIRED,
    FeatureMismatch,
    NoActiveModel,
    NoCapacity,
    QuarantineConflict,
    Registry,
    SwapConflict,
    UnknownVersion,
)


@pytest.fixture(scope="module")
def manager():
    ctx = multiprocessing.get_context("fork")
    mgr = ctx.Manager()
    yield mgr
    mgr.shutdown()


@pytest.fixture()
def registry(manager):
    reg = Registry(manager)
    yield reg
    # Clear state between tests so the shared manager stays pristine.
    for slot in list(reg._slots.keys()):
        reg._slots[slot] = {
            "slot_id": slot,
            "version": None,
            "process_state": "idle",
            "pid": None,
            "in_flight": 0,
            "accepted": 0,
            "completed": 0,
            "abandoned": 0,
            "started_at": None,
            "idle_since": 0.0,
            "generation": reg._slots[slot]["generation"],
            "quarantine_state": QUARANTINE_NONE,
        }
    for version in list(reg._versions.keys()):
        del reg._versions[version]
    for key in list(reg._claims.keys()):
        del reg._claims[key]
    for token in list(reg._work.keys()):
        del reg._work[token]
    reg._state["active_version"] = None


def _load(reg: Registry, version: str) -> str:
    slot = reg.reserve_any_slot(version)
    reg.set_pid(version, 1000)
    return slot


def _activate(reg: Registry, version: str, old: str | None = None, token: str = "t") -> None:
    reg.begin_swap(old, token)
    reg.mark_candidate_tested(version)
    reg.commit_swap(version, old, token)


def test_first_load_claims_none_and_activates(registry):
    slot = _load(registry, "v1")
    assert slot == "slot-0"
    registry.mark_candidate_tested("v1")
    old = registry.begin_swap(None, "tok")
    registry.commit_swap("v1", old, "tok")
    assert registry.active_version() == "v1"
    assert registry.version_info("v1")["state"] == VERSION_ACTIVE
    assert registry.snapshot()["slots_in_use"] == 1


def test_second_swap_retires_old_but_keeps_slot_until_drained(registry):
    _load(registry, "v1")
    _activate(registry, "v1")
    _load(registry, "v2")
    _activate(registry, "v2", old="v1")

    assert registry.active_version() == "v2"
    assert registry.version_info("v1")["state"] == VERSION_RETIRED
    assert registry.snapshot()["slots_in_use"] == 2  # v1 still draining
    assert not registry.has_capacity()

    slot = registry.version_info("v1")["slot_id"]
    # In-flight work blocks release.
    version, _ = registry.accept_work()
    assert version == "v2"
    assert registry.release_slot(slot, "v1", process_exited=True) is True  # no in-flight on v1
    registry.complete_work("v2")


def test_no_capacity_when_both_slots_held(registry):
    _load(registry, "v1")
    _activate(registry, "v1")
    _load(registry, "v2")
    _activate(registry, "v2", old="v1")
    with pytest.raises(NoCapacity):
        registry.reserve_any_slot("v3")
    # The current version is untouched.
    assert registry.active_version() == "v2"


def test_in_flight_request_blocks_slot_release(registry):
    _load(registry, "v1")
    _activate(registry, "v1")
    _load(registry, "v2")
    _activate(registry, "v2", old="v1")

    # A request accepted on v1 *before* the swap is counted on v1.
    # (Simulated here by accepting on v1 while it was active is not possible
    # after commit, so drive the counter the way the API does.)
    registry._slots["slot-0"] = {**registry._slots["slot-0"], "in_flight": 1}
    assert registry.release_slot("slot-0", "v1", process_exited=True) is False
    assert registry.release_slot("slot-0", "v1", process_exited=False) is False
    registry.complete_work("v1")
    assert registry.release_slot("slot-0", "v1", process_exited=True) is True
    assert registry.has_capacity()


def test_concurrent_swap_claim_allows_one_winner(registry):
    _load(registry, "v1")
    _activate(registry, "v1")
    assert registry.begin_swap("v1", "a") == "v1"
    with pytest.raises(SwapConflict, match="already in progress"):
        registry.begin_swap("v1", "b")
    registry.end_swap("v1", "a")
    assert registry.begin_swap("v1", "b") == "v1"


def test_stale_expected_version_is_conflict(registry):
    _load(registry, "v1")
    _activate(registry, "v1")
    with pytest.raises(SwapConflict, match="precondition"):
        registry.begin_swap("v0", "token")


def test_commit_requires_smoke_test(registry):
    _load(registry, "v1")
    registry.begin_swap(None, "t")
    with pytest.raises(SwapConflict, match="smoke test"):
        registry.commit_swap("v1", None, "t")


def test_failed_candidate_frees_slot_and_can_be_retried(registry):
    _load(registry, "v1")
    _activate(registry, "v1")
    registry.begin_swap("v1", "t")
    _load(registry, "bad")
    registry.fail_candidate("bad", "boom")
    registry.end_swap("v1", "t")

    info = registry.version_info("bad")
    assert info["state"] == VERSION_FAILED
    assert info["error"] == "boom"
    assert registry.has_capacity()
    assert registry.active_version() == "v1"

    registry.begin_swap("v1", "t2")
    _load(registry, "bad")  # retry allowed, keeps previous error
    assert registry.version_info("bad")["attempt"] == 2
    registry.end_swap("v1", "t2")


def test_duplicate_live_version_rejected(registry):
    _load(registry, "v1")
    _activate(registry, "v1")
    with pytest.raises(SwapConflict, match="already exists"):
        registry.reserve_any_slot("v1")


def test_accepted_work_is_pinned_to_active_version(registry):
    _load(registry, "v1")
    _activate(registry, "v1")
    version, slot = registry.accept_work()
    assert (version, slot) == ("v1", "slot-0")

    _load(registry, "v2")
    _activate(registry, "v2", old="v1")
    # The pre-existing in-flight count belongs to v1, not the new active.
    assert registry.version_in_flight("v1") == 1
    assert registry.version_in_flight("v2") == 0
    registry.complete_work("v1")
    assert registry.version_in_flight("v1") == 0


def test_feature_mismatch_does_not_count_in_flight(registry):
    _load(registry, "v1")
    registry.set_model_shape("v1", 3, 2)
    _activate(registry, "v1")
    with pytest.raises(FeatureMismatch):
        registry.accept_work(2)
    assert registry.version_in_flight("v1") == 0
    version, _ = registry.accept_work(3)
    assert version == "v1"


def test_inference_without_model_is_rejected(registry):
    with pytest.raises(NoActiveModel):
        registry.accept_work()


def test_dead_active_worker_releases_slot(registry):
    _load(registry, "v1")
    _activate(registry, "v1")
    registry.accept_work()
    registry.settle_dead_worker("v1")
    assert registry.fail_active("v1", "crashed") is True
    assert registry.active_version() is None
    assert registry.has_capacity()
    assert registry.version_info("v1")["state"] == VERSION_FAILED


# ---------------------------------------------------------------------------
# Emergency quarantine
# ---------------------------------------------------------------------------


def _draining_pair(reg: Registry) -> None:
    """v1 active, v2 active, v1 retired+draining with one accepted request."""
    _load(reg, "v1")
    _activate(reg, "v1")
    token = "w-v1"
    reg.accept_work(token=token)
    _load(reg, "v2")
    _activate(reg, "v2", old="v1")
    assert reg.version_info("v1")["state"] == VERSION_RETIRED
    assert reg.version_in_flight("v1") == 1


def test_quarantine_rejects_active_version(registry):
    _load(registry, "v1")
    _activate(registry, "v1")
    with pytest.raises(QuarantineConflict, match="not a retired"):
        registry.begin_quarantine("v1", "operator")
    # Active version untouched and still able to accept work.
    assert registry.active_version() == "v1"
    assert registry.accept_work()[0] == "v1"


def test_quarantine_rejects_unknown_and_loading(registry):
    with pytest.raises(UnknownVersion):
        registry.begin_quarantine("nope", "operator")
    _load(registry, "v1")
    with pytest.raises(QuarantineConflict, match="not a retired"):
        registry.begin_quarantine("v1", "operator")


def test_quarantine_settles_work_once_and_keeps_slot_occupied(registry):
    _draining_pair(registry)
    record = registry.begin_quarantine("v1", "stuck inference")

    assert record["abandoned_work"] == 1
    assert record["already"] is False
    assert record["slot_id"] == "slot-0"
    assert record["in_flight"] == 1  # NOT faked to zero

    # The work item is settled exactly once, as quarantined.
    work = registry.work_outcome("w-v1")
    assert work["settled"] is True
    assert work["outcome"] == "quarantined"
    assert registry.complete_work("v1", "w-v1") is False
    assert registry.version_in_flight("v1") == 1  # counters not double-settled

    # The active version keeps serving.
    assert registry.active_version() == "v2"
    assert registry.accept_work(token="w-v2")[0] == "v2"

    # Even a "zero in-flight" release is refused for a pending quarantine:
    # the slot stays occupied until the process exit is confirmed.
    assert registry.release_slot("slot-0", "v1", process_exited=False) is False

    info = registry.version_info("v1")
    assert info["quarantine_state"] == QUARANTINE_PENDING
    assert info["quarantine_reason"] == "stuck inference"
    assert info["state"] == VERSION_RETIRED
    assert registry.snapshot()["slots_in_use"] == 2

    # Only the confirmed exit frees the slot.
    assert registry.release_slot("slot-0", "v1", process_exited=True) is True
    assert registry.snapshot()["slots_in_use"] == 1
    assert registry.version_info("v1")["quarantine_state"] == QUARANTINE_RELEASED
    assert registry.version_info("v1")["process_state"] == "idle"


def test_quarantine_is_idempotent_for_same_version(registry):
    _draining_pair(registry)
    first = registry.begin_quarantine("v1", "stuck")
    second = registry.begin_quarantine("v1", "again")
    assert second["already"] is True
    assert second["quarantined_at"] == first["quarantined_at"]
    assert second["abandoned_work"] == first["abandoned_work"] == 1
    assert registry.complete_work("v1", "w-v1") is False


def test_late_completion_after_quarantine_does_not_touch_counters(registry):
    _draining_pair(registry)
    registry.begin_quarantine("v1", "stuck")
    # Worker response finally arrives: must be ignored, not counted.
    assert registry.complete_work("v1", "w-v1") is False
    slot = next(s for s in registry.snapshot()["slots"] if s["slot_id"] == "slot-0")
    assert slot["in_flight"] == 1
    assert slot["completed"] == 0
    assert registry.work_outcome("w-v1")["outcome"] == "quarantined"

    # After the process exit is confirmed the slot is reused; another late
    # response (old generation) must not touch the new process either.
    assert registry.release_slot("slot-0", "v1", process_exited=True) is True
    _load(registry, "v3")
    assert registry.complete_work("v1", "w-v1") is False
    new_slot = next(s for s in registry.snapshot()["slots"] if s["slot_id"] == "slot-0")
    assert new_slot["version"] == "v3"
    assert new_slot["in_flight"] == 0
    assert new_slot["completed"] == 0


def test_quarantine_does_not_disturb_other_versions_in_flight(registry):
    _draining_pair(registry)
    registry.accept_work(token="w-v2")
    registry.begin_quarantine("v1", "stuck")
    assert registry.version_in_flight("v2") == 1
    assert registry.complete_work("v2", "w-v2") is True
    assert registry.version_in_flight("v2") == 0


def test_failed_quarantine_restores_drain_gate(registry):
    _draining_pair(registry)
    record = registry.begin_quarantine("v1", "stuck")
    assert record["abandoned_work"] == 1

    result = registry.abort_quarantine("v1", "process did not exit")
    assert result["quarantine_state"] == QUARANTINE_FAILED
    assert result["in_flight"] == 1
    info = registry.version_info("v1")
    assert info["quarantine_state"] == QUARANTINE_FAILED
    assert info["error"] == "process did not exit"

    # The work is pending again, so a normal drain must still wait for the
    # actual completion; the slot is not reported as free.
    assert registry.complete_work("v1", "w-v1") is True
    assert registry.version_in_flight("v1") == 0
    assert registry.release_slot("slot-0", "v1", process_exited=True) is True
    # The failed attempt stays visible in the version history for audit.
    info = registry.version_info("v1")
    assert info["quarantine_state"] == QUARANTINE_FAILED
    assert info["error"] == "process did not exit"


def test_failed_quarantine_accounts_for_late_completion_once(registry):
    _draining_pair(registry)
    registry.begin_quarantine("v1", "stuck")
    # Worker response arrives while the emergency stop is still unconfirmed.
    assert registry.complete_work("v1", "w-v1") is False
    assert registry.work_outcome("w-v1")["late_completion"] is True

    result = registry.abort_quarantine("v1", "process did not exit")
    assert result["in_flight"] == 0
    slot = next(s for s in registry.snapshot()["slots"] if s["slot_id"] == "slot-0")
    assert slot["completed"] == 1
    # A duplicate late response changes nothing.
    assert registry.complete_work("v1", "w-v1") is False
    assert next(
        s for s in registry.snapshot()["slots"] if s["slot_id"] == "slot-0"
    )["completed"] == 1
    assert registry.release_slot("slot-0", "v1", process_exited=True) is True


def test_normal_completion_still_settles_before_quarantine(registry):
    _draining_pair(registry)
    assert registry.complete_work("v1", "w-v1") is True
    record = registry.begin_quarantine("v1", "no work left")
    assert record["abandoned_work"] == 0
    assert record["in_flight"] == 0
    # Completed normally, so the emergency stop can release the slot at once.
    assert registry.release_slot("slot-0", "v1", process_exited=True) is True
