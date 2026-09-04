"""
data/training_job_store.py — File-backed async job store for CSV training
=============================================================================
SRS 5.6.7 — POST /api/ai/train is asynchronous: it validates + saves the
CSV, returns a jobId immediately, and trains in a background thread.
GET /api/ai/train/status/{jobId} polls progress: queued -> preprocessing
-> training -> complete (or failed).

Stage C investigation found no task-queue infrastructure anywhere in this
module (no Celery/RQ/APScheduler/threading already in use) — this store
follows the same file-based pattern as models/model_registry.py rather
than introducing new infrastructure (Redis, a message broker, etc.).

Design note — why ONE FILE PER JOB, not one shared jobs.json:
    The documented production run command is
    `gunicorn -w 2 -b 0.0.0.0:5000 app:app` — two separate worker
    PROCESSES. A GET status poll can land on a different worker process
    than the one running the POST's background thread, so a plain
    in-memory dict would be invisible across workers. A single shared
    jobs.json would dodge that but introduces a cross-process
    read-modify-write race: two different jobs' status updates, each
    landing on a different worker, could race on the same shared file and
    lose an update (worker B's write, based on a snapshot read before
    worker A's write landed, clobbers A's update when it saves).

    Splitting storage to one file per job (`{job_id}.json`) removes the
    shared mutable state entirely: different jobs never touch the same
    file, and a given job's file is only ever written by the single
    background thread that owns that job's lifecycle (created it, and
    is the only thread that ever calls update_status/set_result/set_error
    for that job_id). So there is no cross-process OR cross-thread race
    to guard against — atomic replace (write to a temp file, then
    os.replace) is sufficient to guarantee a concurrent GET never reads a
    half-written file.

    Known accepted limitation: if the process/worker running a job's
    background thread crashes mid-training, that job's file is left
    stuck at whatever status it last reached (e.g. 'training') with no
    separate timeout/recovery mechanism. Given this module's "local
    files, no new infra" pattern and that training is a short-lived,
    admin-triggered action (not a long-running queue), this was judged
    an acceptable trade-off rather than building a recovery system.
"""

import json
import logging
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger('slt_ai.jobs')

BASE_DIR = Path(__file__).resolve().parent.parent
JOBS_DIR = BASE_DIR / 'data' / 'jobs'

VALID_STATUSES = {'queued', 'preprocessing', 'training', 'complete', 'failed'}


class TrainingJobStore:

    def __init__(self):
        JOBS_DIR.mkdir(parents=True, exist_ok=True)

    def create_job(self, upload_summary: dict) -> str:
        job_id = uuid.uuid4().hex
        now = datetime.utcnow().isoformat() + 'Z'
        record = {
            'jobId':         job_id,
            'status':        'queued',
            'createdAt':     now,
            'updatedAt':     now,
            'uploadSummary': upload_summary,
            'result':        None,
            'error':         None,
        }
        self._write(job_id, record)
        logger.info(f"Training job {job_id} created (queued)")
        return job_id

    def update_status(self, job_id: str, status: str) -> None:
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid status: {status}")
        record = self.get(job_id)
        if record is None:
            logger.warning(f"update_status: unknown job {job_id}")
            return
        record['status']    = status
        record['updatedAt'] = datetime.utcnow().isoformat() + 'Z'
        self._write(job_id, record)
        logger.info(f"Training job {job_id} -> {status}")

    def set_result(self, job_id: str, result: dict) -> None:
        record = self.get(job_id)
        if record is None:
            logger.warning(f"set_result: unknown job {job_id}")
            return
        record['status']    = 'complete'
        record['result']    = result
        record['updatedAt'] = datetime.utcnow().isoformat() + 'Z'
        self._write(job_id, record)
        logger.info(f"Training job {job_id} complete")

    def set_error(self, job_id: str, message: str) -> None:
        record = self.get(job_id)
        if record is None:
            logger.warning(f"set_error: unknown job {job_id}")
            return
        record['status']    = 'failed'
        record['error']     = message
        record['updatedAt'] = datetime.utcnow().isoformat() + 'Z'
        self._write(job_id, record)
        logger.warning(f"Training job {job_id} failed: {message}")

    def get(self, job_id: str) -> Optional[dict]:
        path = self._path(job_id)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except Exception as exc:
            logger.warning(f"Job {job_id} file unreadable: {exc}")
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE
    # ─────────────────────────────────────────────────────────────────────────

    def _path(self, job_id: str) -> Path:
        return JOBS_DIR / f'{job_id}.json'

    def _write(self, job_id: str, record: dict) -> None:
        """Atomic write — temp file + os.replace, so a concurrent GET never
        sees a half-written file."""
        fd, tmp_path = tempfile.mkstemp(dir=JOBS_DIR, prefix=f'.{job_id}.', suffix='.tmp')
        try:
            with os.fdopen(fd, 'w') as f:
                json.dump(record, f, indent=2)
            os.replace(tmp_path, self._path(job_id))
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise
