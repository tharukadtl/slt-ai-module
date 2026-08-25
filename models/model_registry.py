"""
models/model_registry.py — File-based model version registry
================================================================
SRS 5.6.7 governance for the CSV training pipeline: retraining must produce
a CANDIDATE version (never silently overwrite the model currently serving
live traffic), report a metrics comparison against the active version, and
require an explicit "Activate Model" step before a candidate goes live.
Rollback must be able to restore a previous active version.

This module is intentionally file-based, not DB-backed — the AI module has
no database tables of its own (see utils/db_connector.py: the MySQL
connection is treated as an optional, gracefully-degradable *data source*,
never as storage for the module's own state), and existing model
persistence (forecasting.py, clustering.py) already uses local pickle
files. A registry.json index + one pickle file per version keeps that same
pattern instead of introducing a new storage mechanism.

Layout (per model, e.g. "forecaster" or "clusterer"):
    {MODEL_DIR}/versions/{model_name}/registry.json   — version index
    {MODEL_DIR}/versions/{model_name}/v{N}.pkl         — fitted object

Each registry entry: {versionId, createdAt, status, metrics, meta}
Status is one of: "candidate" | "active" | "archived".
Exactly one version may be "active" at a time.

Usage:
    from models.model_registry import ModelVersionRegistry, build_comparison

    reg = ModelVersionRegistry(Config.MODEL_DIR, 'forecaster')
    candidate = reg.save_version(fitted_model, metrics, meta, status='candidate')
    comparison = build_comparison(reg.get_active(), candidate,
                                   lower_is_better=['mae', 'rmse'],
                                   higher_is_better=['accuracy'])
    reg.activate(candidate['versionId'])
    reg.rollback()  # or reg.rollback(version_id=3)
"""

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger('slt_ai.model_registry')

DEFAULT_KEEP_LAST_ARCHIVED = 5  # how many archived versions to retain per model


class ModelVersionRegistry:
    """Versioned file-based storage for a single model (one instance per model name)."""

    def __init__(self, model_dir, model_name: str, keep_last_archived: int = DEFAULT_KEEP_LAST_ARCHIVED):
        self.model_name = model_name
        self.dir = Path(model_dir) / 'versions' / model_name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.dir / 'registry.json'
        self.keep_last_archived = keep_last_archived
        self._data = self._load_index()

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────────────────────

    def list_versions(self) -> list:
        """All versions, newest first."""
        return sorted(self._data['versions'], key=lambda v: v['versionId'], reverse=True)

    def get_active(self) -> Optional[dict]:
        for v in self._data['versions']:
            if v['status'] == 'active':
                return v
        return None

    def get_version(self, version_id: int) -> Optional[dict]:
        return self._find(version_id)

    def save_version(self, obj, metrics: dict, meta: dict, status: str = 'candidate') -> dict:
        """
        Persist a newly fitted object as a new version.

        status='candidate' — the common governance path: saved alongside the
            active version, does NOT touch what's currently serving traffic.
        status='active'    — used only by callers that must bypass the gate
            (e.g. forecasting.py's internal freshness auto-refit, which is
            unrelated to the admin CSV-training governance flow — see that
            module for the reasoning). Archives whatever was active before.
        """
        if status not in ('candidate', 'active', 'archived'):
            raise ValueError(f"Invalid status: {status}")

        version_id = self._data['nextId']
        self._data['nextId'] += 1

        with open(self._version_path(version_id), 'wb') as f:
            pickle.dump(obj, f)

        entry = {
            'versionId': version_id,
            'createdAt': datetime.utcnow().isoformat() + 'Z',
            'status':    status,
            'metrics':   metrics,
            'meta':      meta,
        }
        if status == 'active':
            self._archive_current_active()
        self._data['versions'].append(entry)
        self._save_index()
        self._prune_archived()
        logger.info(f"[{self.model_name}] saved version {version_id} (status={status})")
        return entry

    def activate(self, version_id: int) -> dict:
        """Explicit promotion step — the only way a candidate becomes active."""
        target = self._find(version_id)
        if target is None:
            raise ValueError(f"Version {version_id} not found for model '{self.model_name}'")
        if target['status'] == 'active':
            return target

        self._archive_current_active()
        target['status'] = 'active'
        self._save_index()
        self._prune_archived()
        logger.info(f"[{self.model_name}] activated version {version_id}")
        return target

    def rollback(self, version_id: Optional[int] = None) -> dict:
        """
        Revert the active pointer.

        version_id=None: rolls back to the most recently archived version
        (i.e. whatever was active immediately before the current one).
        """
        if version_id is not None:
            return self.activate(version_id)

        archived = [v for v in self._data['versions'] if v['status'] == 'archived']
        if not archived:
            raise ValueError(f"No previous version to roll back to for model '{self.model_name}'")
        target = max(archived, key=lambda v: v['versionId'])
        return self.activate(target['versionId'])

    def load_object(self, version_id: int):
        """Load the pickled fitted object for a version."""
        path = self._version_path(version_id)
        if not path.exists():
            raise FileNotFoundError(f"Version {version_id} pickle missing for '{self.model_name}'")
        with open(path, 'rb') as f:
            return pickle.load(f)

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE
    # ─────────────────────────────────────────────────────────────────────────

    def _load_index(self) -> dict:
        if self.registry_path.exists():
            try:
                return json.loads(self.registry_path.read_text())
            except Exception as exc:
                logger.warning(f"[{self.model_name}] registry.json unreadable ({exc}) — starting fresh")
        return {'nextId': 1, 'versions': []}

    def _save_index(self) -> None:
        self.registry_path.write_text(json.dumps(self._data, indent=2))

    def _version_path(self, version_id: int) -> Path:
        return self.dir / f'v{version_id}.pkl'

    def _find(self, version_id: int) -> Optional[dict]:
        for v in self._data['versions']:
            if v['versionId'] == version_id:
                return v
        return None

    def _archive_current_active(self) -> None:
        for v in self._data['versions']:
            if v['status'] == 'active':
                v['status'] = 'archived'

    def _prune_archived(self) -> None:
        """Keep only the most recent `keep_last_archived` archived versions."""
        archived = sorted(
            (v for v in self._data['versions'] if v['status'] == 'archived'),
            key=lambda v: v['versionId'],
            reverse=True,
        )
        stale = archived[self.keep_last_archived:]
        if not stale:
            return
        stale_ids = {v['versionId'] for v in stale}
        for v in stale:
            path = self._version_path(v['versionId'])
            if path.exists():
                path.unlink()
        self._data['versions'] = [v for v in self._data['versions'] if v['versionId'] not in stale_ids]
        self._save_index()
        logger.info(f"[{self.model_name}] pruned {len(stale_ids)} archived version(s): {sorted(stale_ids)}")


def build_comparison(previous: Optional[dict], candidate: dict,
                      lower_is_better: list, higher_is_better: list) -> dict:
    """
    Build a metrics comparison object: candidate vs. the currently active version.

    previous: registry entry dict (or None if there's no active version yet —
              e.g. the very first training run).
    candidate: registry entry dict for the newly saved candidate.
    """
    if previous is None:
        return {
            'previousVersionId': None,
            'delta': None,
            'note': 'No active model to compare against — this is the first trained version.',
        }

    prev_metrics = previous.get('metrics') or {}
    cand_metrics = candidate.get('metrics') or {}
    delta = {}
    for key in [*lower_is_better, *higher_is_better]:
        prev_v = prev_metrics.get(key)
        new_v  = cand_metrics.get(key)
        if prev_v is None or new_v is None:
            delta[key] = None
            continue
        change = round(new_v - prev_v, 4)
        improved = (change < 0) if key in lower_is_better else (change > 0)
        delta[key] = {
            'previous':  prev_v,
            'candidate': new_v,
            'change':    change,
            'improved':  improved,
        }

    return {
        'previousVersionId': previous.get('versionId'),
        'delta': delta,
    }
