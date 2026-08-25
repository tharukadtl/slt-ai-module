"""
tests/test_model_registry.py — Model Version Retention (AI-025, FR-30) Unit Tests
==================================================================================
Sheet 10_AI_MODULE row AI-025 asks that repeated retrain+activate cycles do not
grow the on-disk model registry without bound: after nine cycles only the active
version plus the last five archived ones should remain — six in total — and the
pruned versions' ``.pkl`` files must actually be gone from disk, not merely
dropped from the index.

``models/model_registry.py``'s ``_prune_archived()`` is the code under test.
``DEFAULT_KEEP_LAST_ARCHIVED = 5``, so "6 total" is 1 active + 5 archived, and the
row's number is asserted against that constant rather than hard-coded, so this
test explains itself if the retention policy is ever deliberately changed.

Every test here runs against a ``tmp_path`` registry root. The real registry
(``models/saved/versions/<model>/``) is what the running Flask service loads at
boot, and a retention test by definition deletes versions — pointing it at the
real directory would destroy live model state.

The stored objects are small dicts rather than fitted Prophet/K-Means models:
the registry pickles whatever it is handed and never inspects it, so retention
behaviour is identical and nine real fits are not needed.

Run:
    cd slt-ai-module
    venv/Scripts/python.exe -m pytest tests/test_model_registry.py -v
"""

import os
import sys
import json

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_registry import (
    ModelVersionRegistry,
    DEFAULT_KEEP_LAST_ARCHIVED,
    build_comparison,
)

CYCLES = 9
EXPECTED_TOTAL = 1 + DEFAULT_KEEP_LAST_ARCHIVED    # 1 active + 5 archived = 6


@pytest.fixture
def registry(tmp_path):
    """A throwaway registry rooted in tmp_path — never the real models/saved."""
    return ModelVersionRegistry(tmp_path, 'forecaster')


def test_retention_prunes_to_six(registry, tmp_path):
    """AI-025 — nine retrain+activate cycles leave exactly active + 5 archived."""
    version_ids = []
    for cycle in range(1, CYCLES + 1):
        # Each cycle mirrors the real flow: retrain() saves a CANDIDATE, then the
        # admin's explicit Activate Model promotes it and archives the previous.
        candidate = registry.save_version(
            {'stand-in': f'fitted model from cycle {cycle}'},
            {'mae': 10.0 - cycle, 'rmse': 12.0 - cycle, 'accuracy': 60.0 + cycle},
            {'training_rows': 200 + cycle},
            status='candidate',
        )
        registry.activate(candidate['versionId'])
        version_ids.append(candidate['versionId'])

    versions = registry.list_versions()
    by_status = {}
    for v in versions:
        by_status.setdefault(v['status'], []).append(v['versionId'])

    # ── Step 2: exactly one active + KEEP_LAST_ARCHIVED archived ─────────────
    assert len(versions) == EXPECTED_TOTAL, (
        f"{len(versions)} versions retained after {CYCLES} cycles, expected "
        f"{EXPECTED_TOTAL}: {json.dumps(by_status, sort_keys=True)}"
    )
    assert by_status.get('active') == [version_ids[-1]], (
        f"The most recently activated version {version_ids[-1]} should be the "
        f"only active one, got {by_status.get('active')}"
    )
    assert sorted(by_status.get('archived', [])) == sorted(version_ids[-1 - DEFAULT_KEEP_LAST_ARCHIVED:-1]), (
        "The retained archived versions should be the five immediately preceding "
        f"the active one, got {sorted(by_status.get('archived', []))}"
    )
    assert 'candidate' not in by_status, \
        "Every candidate was activated, so none should remain in candidate state"

    # ── Step 3: the pruned pickles are actually deleted from disk ────────────
    version_dir = tmp_path / 'versions' / 'forecaster'
    pkls = sorted(p.name for p in version_dir.glob('*.pkl'))
    assert len(pkls) == EXPECTED_TOTAL, (
        f"{len(pkls)} .pkl files left on disk, expected {EXPECTED_TOTAL} — "
        f"pruning removed the index entries but not the files: {pkls}"
    )
    retained_ids = {v['versionId'] for v in versions}
    assert pkls == sorted(f'v{i}.pkl' for i in retained_ids), (
        f"On-disk files {pkls} do not match the retained index entries "
        f"{sorted(retained_ids)}"
    )

    # And the survivors are still loadable — pruning did not corrupt them.
    for vid in retained_ids:
        assert registry.load_object(vid)['stand-in'].endswith(f'cycle {vid}')


def test_pruning_never_touches_the_active_version(registry):
    """
    The retention policy counts archived versions only. The active one must
    survive regardless of how many cycles run, since it is what serves traffic.
    """
    for cycle in range(1, CYCLES + 1):
        entry = registry.save_version(
            {'stand-in': cycle}, {'mae': 1.0}, {}, status='candidate',
        )
        registry.activate(entry['versionId'])
        active = registry.get_active()
        assert active is not None, f"No active version after cycle {cycle}"
        assert active['versionId'] == entry['versionId']


def test_candidates_are_not_pruned_by_retention(registry):
    """
    Documents the real policy boundary, so a future change is a deliberate one:
    ``_prune_archived`` filters on ``status == 'archived'`` only, so candidates
    that are never activated accumulate indefinitely. Nine unactivated
    candidates therefore all survive — the opposite of the archived case above.
    """
    for cycle in range(CYCLES):
        registry.save_version({'stand-in': cycle}, {'mae': 1.0}, {}, status='candidate')

    versions = registry.list_versions()
    assert len(versions) == CYCLES
    assert all(v['status'] == 'candidate' for v in versions)


def test_rollback_restores_the_previously_active_version(registry):
    """
    Retention has to leave rollback usable: after pruning, the most recent
    archived version is still present and ``rollback()`` promotes it back.
    """
    ids = []
    for cycle in range(1, CYCLES + 1):
        entry = registry.save_version({'stand-in': cycle}, {'mae': 1.0}, {}, status='candidate')
        registry.activate(entry['versionId'])
        ids.append(entry['versionId'])

    restored = registry.rollback()
    assert restored['versionId'] == ids[-2], (
        f"rollback() promoted v{restored['versionId']}, expected the "
        f"immediately-previous active version v{ids[-2]}"
    )
    assert registry.get_active()['versionId'] == ids[-2]


def test_build_comparison_reports_direction_of_change(registry):
    """
    The comparison object the Model Training UI renders must say which way each
    metric moved, and get the direction right for both lower-is-better (mae)
    and higher-is-better (accuracy) metrics.
    """
    previous = registry.save_version(
        {'stand-in': 'prev'}, {'mae': 5.0, 'accuracy': 70.0}, {}, status='active',
    )
    candidate = registry.save_version(
        {'stand-in': 'cand'}, {'mae': 3.0, 'accuracy': 65.0}, {}, status='candidate',
    )

    comparison = build_comparison(
        previous, candidate, lower_is_better=['mae'], higher_is_better=['accuracy'],
    )
    assert comparison['previousVersionId'] == previous['versionId']
    assert comparison['delta']['mae']['change'] == -2.0
    assert comparison['delta']['mae']['improved'] is True
    assert comparison['delta']['accuracy']['change'] == -5.0
    assert comparison['delta']['accuracy']['improved'] is False
