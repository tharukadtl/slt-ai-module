"""
tests/test_job_store.py — Async Training Job Store (AI-026 / AI-027, FR-30) Unit Tests
=======================================================================================
Sheet 10_AI_MODULE rows AI-026 and AI-027 describe a startup reconciliation step for
``data/training_job_store.py``:

  AI-026 — a job left in ``training`` with a stale ``updatedAt`` (older than the
           staleness window) must be resolved to ``failed`` with a message saying it
           was interrupted by a service restart, instead of leaving an admin's
           status-polling tab spinning forever.
  AI-027 — a job whose ``updatedAt`` is recent must be left completely untouched.
           The documented production run command is ``gunicorn -w 2``, i.e. two
           worker PROCESSES, so a restarting worker must not be able to kill a job
           that another, still-running worker is actively executing. The staleness
           window is precisely the guard against that race.

Current state of the implementation, verified by search rather than assumed:
``reconcile_stuck_jobs`` does not exist anywhere in this repository — not as a
module-level function in ``data/training_job_store.py``, not as a method on
``TrainingJobStore``, and not as a startup hook in ``app.py``. The module's own
docstring records this as a deliberate, accepted trade-off:

    "Known accepted limitation: if the process/worker running a job's background
     thread crashes mid-training, that job's file is left stuck at whatever status
     it last reached (e.g. 'training') with no separate timeout/recovery
     mechanism ... this was judged an acceptable trade-off rather than building a
     recovery system."

So these two rows encode a requirement the code deliberately does not meet. Both
tests below therefore locate the reconciler first and fail with that explanation
if it is absent — a red result on a real, documented gap rather than a collection
error. If the reconciler is ever added, both tests immediately go on to exercise
the real behaviour without needing to be rewritten.

Every test runs against a ``tmp_path`` jobs directory (``JOBS_DIR`` is monkeypatched)
so no real queued/running job file is ever read, rewritten or failed by this suite.

Run:
    cd slt-ai-module
    venv/Scripts/python.exe -m pytest tests/test_job_store.py -v
"""

import os
import sys
import json
from datetime import datetime, timedelta

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import training_job_store as job_store_mod
from data.training_job_store import TrainingJobStore


# The row's own threshold: "updatedAt (>30min)" is stale, "<30min" is live.
STALENESS_MINUTES = 30


@pytest.fixture
def store(monkeypatch, tmp_path):
    """A TrainingJobStore backed by a throwaway jobs directory."""
    jobs_dir = tmp_path / 'jobs'
    jobs_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(job_store_mod, 'JOBS_DIR', jobs_dir)
    return TrainingJobStore()


def _resolve_reconciler(store):
    """
    Return a zero-argument callable that runs startup reconciliation, or None.

    Looks in the three places the feature could reasonably live, so this test
    does not fail merely because the implementation chose a different shape:
    a bound method on the store, a module-level function taking the store, or a
    module-level function taking nothing.
    """
    method = getattr(store, 'reconcile_stuck_jobs', None)
    if callable(method):
        return method

    func = getattr(job_store_mod, 'reconcile_stuck_jobs', None)
    if callable(func):
        try:
            import inspect
            takes_arg = len(inspect.signature(func).parameters) >= 1
        except (TypeError, ValueError):
            takes_arg = False
        return (lambda: func(store)) if takes_arg else func

    return None


_MISSING = (
    "No startup reconciliation exists. `reconcile_stuck_jobs` is not defined on "
    "TrainingJobStore, in data/training_job_store.py, or in app.py — the module's "
    "own docstring records the stuck-job case as a knowingly accepted limitation "
    "('no separate timeout/recovery mechanism'). A job whose worker dies mid-training "
    "therefore stays at status='training' permanently and GET /api/ai/train/status/"
    "{jobId} keeps reporting it as in progress forever."
)


def _seed_job(store, minutes_ago: int, status: str = 'training') -> str:
    """Create a job and force its status/updatedAt to a chosen point in time."""
    job_id = store.create_job({'filename': 'faults.csv', 'rowCount': 120})
    record = store.get(job_id)
    record['status']    = status
    record['updatedAt'] = (datetime.utcnow() - timedelta(minutes=minutes_ago)).isoformat() + 'Z'
    store._write(job_id, record)
    return job_id


def test_reconciliation_marks_stale_jobs_failed(store):
    """AI-026 — a job stuck in 'training' past the staleness window resolves to failed."""
    job_id = _seed_job(store, minutes_ago=STALENESS_MINUTES + 15)

    # Pre-condition: the crash is simulated, and the job really is stuck.
    assert store.get(job_id)['status'] == 'training'

    reconcile = _resolve_reconciler(store)
    assert reconcile is not None, _MISSING

    reconcile()

    job = store.get(job_id)
    assert job['status'] == 'failed', (
        f"Stale job left at status={job['status']!r} — an admin polling "
        "GET /api/ai/train/status/{jobId} would wait indefinitely"
    )
    message = f"{job.get('error') or ''} {job.get('message') or ''}"
    assert 'restart' in message.lower() or 'interrupt' in message.lower(), (
        f"Failure message does not explain what happened: {message!r}"
    )


def test_reconciliation_ignores_fresh_jobs(store):
    """
    AI-027 — a job updated moments ago is genuinely alive (another gunicorn worker
    is running it) and must be left exactly as it is.
    """
    fresh_id = _seed_job(store, minutes_ago=1)
    before = store.get(fresh_id)

    reconcile = _resolve_reconciler(store)
    assert reconcile is not None, _MISSING

    reconcile()

    after = store.get(fresh_id)
    assert after['status'] == 'training', (
        f"A live job was moved to {after['status']!r} — one worker's restart just "
        "killed another worker's running job"
    )
    assert after['updatedAt'] == before['updatedAt'], \
        "A live job's updatedAt was rewritten by reconciliation"
    assert after.get('error') is None


def test_reconciliation_leaves_terminal_jobs_alone(store):
    """
    The boundary the two rows imply between them: reconciliation is about jobs
    still claiming to be in progress. A long-finished 'complete' job is stale by
    timestamp but must not be rewritten to 'failed'.
    """
    done_id = _seed_job(store, minutes_ago=STALENESS_MINUTES + 120, status='complete')

    reconcile = _resolve_reconciler(store)
    assert reconcile is not None, _MISSING

    reconcile()
    assert store.get(done_id)['status'] == 'complete'


# ═════════════════════════════════════════════════════════════════════════════
# BASELINE — the job store's existing, implemented behaviour
# ═════════════════════════════════════════════════════════════════════════════
# These pass today and localise the three reds above: the store itself is sound,
# it is specifically the recovery path that was never built.

class TestJobStoreBasics:

    def test_create_job_starts_queued(self, store):
        job_id = store.create_job({'rowCount': 4})
        job = store.get(job_id)
        assert job['status'] == 'queued'
        assert job['jobId'] == job_id
        assert job['result'] is None and job['error'] is None

    def test_status_transitions_are_persisted(self, store):
        job_id = store.create_job({'rowCount': 4})
        for status in ('preprocessing', 'training'):
            store.update_status(job_id, status)
            assert store.get(job_id)['status'] == status

    def test_invalid_status_is_rejected(self, store):
        job_id = store.create_job({'rowCount': 4})
        with pytest.raises(ValueError):
            store.update_status(job_id, 'nonsense')

    def test_set_result_completes_the_job(self, store):
        job_id = store.create_job({'rowCount': 4})
        store.set_result(job_id, {'forecaster': {'status': 'candidate'}})
        job = store.get(job_id)
        assert job['status'] == 'complete'
        assert job['result']['forecaster']['status'] == 'candidate'

    def test_set_error_fails_the_job(self, store):
        job_id = store.create_job({'rowCount': 4})
        store.set_error(job_id, 'boom')
        job = store.get(job_id)
        assert job['status'] == 'failed'
        assert job['error'] == 'boom'

    def test_unknown_job_returns_none(self, store):
        assert store.get('does-not-exist') is None

    def test_each_job_gets_its_own_file(self, store, tmp_path):
        """One file per job — the cross-process design the module documents."""
        ids = [store.create_job({'rowCount': i}) for i in range(3)]
        files = sorted(p.name for p in (tmp_path / 'jobs').glob('*.json'))
        assert files == sorted(f'{i}.json' for i in ids)
        for job_id in ids:
            assert json.loads((tmp_path / 'jobs' / f'{job_id}.json').read_text())['jobId'] == job_id
