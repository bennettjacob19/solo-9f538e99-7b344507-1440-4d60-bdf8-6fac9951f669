# Edge QC Classifier Inference Service

A FastAPI + NumPy inference service for an edge quality-control machine with a
hard constraint: **the machine can host at most two model processes**. Models
are replaced online — without dropping accepted inference and without ever
running a third model process.

```
POST /infer           classify a feature vector (with optional client timeout)
POST /admin/models    upload weights/bias/labels and hot-swap the model in
POST /admin/models/{version}/quarantine
                      emergency-isolate a draining version
GET  /admin/versions  version registry
GET  /admin/status    active version, in-flight counts, per-process state
GET  /healthz         liveness
GET  /docs            Swagger UI (interactive; shows every field)
```

## Running it

```bash
docker compose up --build
# API + Swagger UI:  http://localhost:8000/docs
# quick demo:        python examples/demo.py http://localhost:8000
```

Or without Docker:

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
DATA_DIR=./data uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Tests (they start real API and model processes):

```bash
pip install -r requirements-dev.txt
python -m pytest
```

## Model artifact

Each version is a linear classifier. 120 draws a class per row: `score =
weights @ features + bias`, and the highest-scoring label wins.

```json
{
  "version": "v2",
  "weights": [[1.2, -0.4, 0.0, 0.3],
              [-0.2, 0.9, 0.5, -0.1],
              [0.1, -0.8, 1.1, 0.2]],
  "bias": [-0.10, 0.05, -0.20],
  "labels": ["pass", "scratch", "dent"]
}
```

`weights` must be `n_classes x n_features`, `bias` must have `n_classes`
entries, and `labels` must be `n_classes` unique non-empty strings. Anything
else is rejected with 422 before a process is started.

`POST /admin/models` also accepts these optional fields:

| field | purpose |
| --- | --- |
| `test_features` | vector for the pre-activation smoke test (defaults to zeros) |
| `expected_active_version` | optimistic guard; 409 if the active version changed |
| `slow_load_ms` | demo: delay worker readiness to watch the old version serve |
| `simulate_load_failure` | demo: make the new worker report a load failure |

## Emergency quarantine of a draining version

When an old version's inference wedges, the machine can be left with both
slots occupied and the next version unable to load. `POST
/admin/models/{version}/quarantine` is the emergency override:

```json
{"reason": "stuck inference", "simulate_stop_failure": false}
```

* **Only a draining version.** The version must already have been retired by a
  successful swap (`state: "retired"`, slot `draining`). The active version
  returns **409 `not_quarantinable`** and keeps serving; unknown ids return
  **404**. A still-loading candidate cannot be quarantined either.
* **Outstanding work fails explicitly.** Every request accepted by that version
  and not yet returned is settled as `quarantined` and its caller gets **409
  `request_quarantined`** (with `version` and `reason`). It no longer hangs
  until some client timeout.
* **Exactly-once settlement.** Each request has a settlement token, recorded
  with the slot generation it was admitted under. A worker response that
  arrives after the quarantine is a *late completion*: it never settles the
  request twice, and if the slot has been reused by a newer version the
  generation check makes it a no-op, so the new process's counters are never
  touched. A concurrent normal completion and quarantine race also settles
  once: the token decides.
* **The slot stays occupied until the process is really gone.** The quarantine
  sends SIGTERM/SIGKILL to the old process group and waits for the actual exit
  (bounded by `QUARANTINE_STOP_SECONDS`). Only then is the slot released; a
  status query in between shows `quarantine_state: "pending"` and the slot
  still in use. The next version can only be loaded after that exit.
* **Failure is never reported as free capacity.** With
  `simulate_stop_failure: true` (or if the process cannot be confirmed gone)
  the endpoint returns **503 `quarantine_stop_failed`**, the provisional
  abandonment is withdrawn, the slot stays occupied, and the normal drain keeps
  waiting for the in-flight work's real completion. Requests whose worker
  response arrives during the failed window still complete normally (200),
  counted once.
* **Normal drain is unchanged.** If the work finishes on its own, the drain
  loop releases the slot exactly as before; the quarantine is then a no-op
  (`already: true`, `abandoned_work: 0`).
* **Status shows the outcome.** `GET /admin/status` reports
  `quarantined_versions` (`state`: `pending` / `released` / `failed`, reason,
  timestamp, abandoned count), per-slot `quarantine_state` / `abandoned`, and
  `total_abandoned`.

## The guarantees and how they are enforced

Each version runs in **its own OS process** (`python -m app.worker_launcher`,
its own process group, private loopback HTTP port). The API process is the only
owner of lifecycle decisions and keeps the registry in a
`multiprocessing.Manager`, so HTTP concurrency cannot race the accounting.
Two fixed slots, `slot-0` and `slot-1`, are the two-process budget.

```
(vacant) --reserve--> loading --commit--> active --retire--> draining --exit--> idle
                           └── load/test failed ──────────────> failed (slot freed)
```

* **At most two model processes.** A slot is reserved only when it is `idle`.
  It becomes `idle` only after the worker's OS process has exited *and* the
  slot's in-flight counter is zero. A `loading` candidate and the `active`
  version both occupy slots, so a third load gets 429 while a swap is in
  progress. The API refuses to start twice against the same data directory
  (file lock), so there is only ever one budget owner.
* **The old version serves during load.** Promotion is one atomic registry
  operation (`commit_swap`) performed only after the candidate answers a
  smoke-test inference. Until then traffic goes to the old process. A failed
  load or smoke test stops the candidate, frees its slot, and returns 502 with
  `fallback_version` = the still-active old version.
* **Accepted inference is pinned to its version.** `accept_work` reads the
  active version *and* increments that slot's in-flight counter in one lock.
  The returned version is used for the whole request. A swap commits
  afterwards, the request still executes on the process it was accepted for.
* **Timeouts do not end the work.** `timeout_ms` (or its seconds alias
  `timeout_seconds`, or the API's default) bounds only how long the caller
  waits. On expiry the client gets 504, but the worker call runs in a task
  shielded from the request timeout/cancellation, and the in-flight counter is
  decremented when the worker actually returns. A disconnected client behaves
  the same way.
* **The old process exits only after real work ends.** The drain loop stops and
  reaps a `draining` process only when its in-flight count is zero. Just after
  a 504 you will see the old version `draining` with `in_flight: 1`; the slot
  is not reusable yet.
* **A full machine refuses the third version, without touching the current
  one.** While both slots are occupied, uploads get 429 (after a short bounded
  wait for a drain that is already finishing). The active version is never
  stopped to make room.
* **Concurrent replacement of the same old version: at most one wins.** A swap
  claims its old version in the registry before starting a process. A second
  swap from the same old version gets 409 immediately and never spawns a
  process. A stale `expected_active_version` also yields 409.
* **Crashed workers are reclaimed.** If a worker dies unexpectedly, its slot
  is released and, if it was active, the service reports 503 (no active model)
  rather than routing to a dead process.
## Demo scenarios

1. **Normal swap.** Upload `examples/models/v1.json`, infer, upload `v2`,
   infer. `/admin/status` shows the previous version `draining` then `idle`.
2. **Load failure.** Upload with `"simulate_load_failure": true` (or send an
   inconsistent artifact). You get 502/422; the old version keeps answering,
   and the candidate's slot is immediately free again.
3. **Slow inference + timeout + capacity.** `POST /infer` with
   `{"features": [...], "slow_ms": 6000, "timeout_ms": 150}`. You get 504.
   `/admin/status` still shows `in_flight: 1`. Upload a new version — it
   becomes active and the slow version goes `draining`. A third upload returns
   429 while the slow computation still runs. After it finishes the slot is
   released and the upload succeeds.
4. **Emergency quarantine.** While that version is `draining` with
   `in_flight: 1`, `POST /admin/models/<old>/quarantine` returns 200 with
   `abandoned_work: 1` and frees the slot after the process exits; the slow
   caller gets 409 `request_quarantined`, and the next version loads. Add
   `"simulate_stop_failure": true` to see the 503 path: the slot stays
   occupied, the active version keeps serving, and the slow caller still
   completes normally.

`python examples/demo.py <base-url>` runs all four end to end.

## Observations from the smoke runs

- Single request latency on loopback: ~5–7 ms end to end (worker computation
  itself is microseconds; this includes HTTP round trips).
- After a 504, the drain gate held the slot for the full remaining computation
  (verified: 429 until the slow request finished, then `idle`).
- SIGTERM with an in-flight request: the request completed with 200, the API
  exited 0/-15 cleanly, and no worker processes remained.

## Layout

```
app/
  main.py             API, swap orchestration, drain loop, status
  process_manager.py  shared registry/state machine + worker process handles
  worker.py           per-versionHTTP server (one process per version)
  worker_launcher.py  process entry point: load artifact, bind, serve, handshake
  models.py           artifact validation, NumPy scoring
tests/
  test_models.py          artifact validation + scoring
  test_registry.py        state machine, claims, drain gate, crash reclaim,
                          quarantine settlement / generation guards
  test_api_integration.py real API + real worker processes
examples/
  demo.py, models/*.json
```

## Configuration

| env var | default | meaning |
| --- | --- | --- |
| `DATA_DIR` | `/data` | models, staging, worker control dirs, instance lock |
| `DEFAULT_INFER_TIMEOUT_MS` | `5000` | client wait when `timeout_ms` is omitted |
| `SWAP_WAIT_SECONDS` | `3` | how long an upload waits for a draining slot before 429 |
| `QUARANTINE_STOP_SECONDS` | `10` | how long an emergency quarantine waits for the old process to exit |
| `SHUTDOWN_GRACE_SECONDS` | `20` | how long shutdown waits for in-flight work |
| `BOOTSTRAP_VERSION` / `BOOTSTRAP_MODEL` | unset | optionally load one artifact at startup |
| `LOG_LEVEL` | `INFO` | logging |
