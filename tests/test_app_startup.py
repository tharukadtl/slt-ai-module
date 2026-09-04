"""
tests/test_app_startup.py — Flask Boot Resilience (AI-028, FR-30) Integration Test
===================================================================================
Sheet 10_AI_MODULE row AI-028: if the startup job-reconciliation step
(``reconcile_stuck_jobs``, the subject of AI-026/AI-027) raises, the AI service must
still boot and serve traffic. A bug in recovery logic should degrade to today's
status quo — some jobs stay stuck — not take the whole AI module offline.

Current state, verified by search rather than assumed: ``app.py`` has no startup
reconciliation hook at all, and ``reconcile_stuck_jobs`` does not exist anywhere in
the repository (see tests/test_job_store.py's header for the full finding and the
``data/training_job_store.py`` docstring that records the stuck-job case as an
accepted limitation). There is consequently no failure mode to make resilient yet.

This test is written so the red result says exactly that, while still proving the
half of the row that *is* checkable today — that the app boots and
``GET /api/ai/health`` answers 200 — so a future implementation only has to make the
hook exist for the whole row to go green. Both halves are evaluated and reported
together rather than aborting on the first, matching the style used in
tests/test_fault_validators.py and tests/test_formatters.py.

Run:
    cd slt-ai-module
    venv/Scripts/python.exe -m pytest tests/test_app_startup.py -v
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


HOOK_NAMES = ('reconcile_stuck_jobs', '_reconcile_stuck_jobs')


def _app_module():
    """
    Import app.py lazily, never at module scope.

    app.py does `from config import ... is_db_available`, which binds the name at
    import time — so whichever test file imports it FIRST decides whether the
    whole session runs against a real MySQL server or conftest.py's synthetic
    fallback. Importing it at collection time here (before conftest's session-scoped
    mock_db_unavailable fixture has run) silently flipped 9 other tests in
    test_routes.py / test_resource_planner.py from synthetic to live-database mode.
    Deferring the import into the test body keeps that fixture authoritative.
    """
    import app as app_mod
    return app_mod


def _find_startup_hook():
    """
    Return (owner, attribute_name) for the startup reconciliation hook, or None.
    Checked on app.py first (where a startup hook would be invoked) and then on
    the job store / its module, so a reasonable alternative placement still counts.
    """
    from data import training_job_store as job_store_mod
    app_mod = _app_module()

    for owner in (app_mod, job_store_mod, app_mod._jobs, type(app_mod._jobs)):
        for name in HOOK_NAMES:
            if callable(getattr(owner, name, None)):
                return owner, name
    return None


def test_reconciliation_failure_does_not_block_boot(monkeypatch):
    """AI-028 — a deliberately broken reconciliation must not stop the app serving."""
    failures = []

    def check(condition, message):
        if not condition:
            failures.append(message)

    # ── Steps 1-2: patch the reconciler to raise, then boot ──────────────────
    hook = _find_startup_hook()
    check(
        hook is not None,
        "No startup reconciliation hook exists to break. `reconcile_stuck_jobs` is "
        "not defined on app.py, data/training_job_store.py, or TrainingJobStore, and "
        "app.py runs nothing at import time between its singleton construction and "
        "its route definitions. The resilience this row asks for cannot be exercised "
        "because the feature it protects has not been built (see AI-026/AI-027 and "
        "data/training_job_store.py's own 'accepted limitation' docstring).",
    )

    if hook is not None:
        owner, name = hook

        def explode(*_args, **_kwargs):
            raise RuntimeError("deliberately broken reconciliation (AI-028)")

        monkeypatch.setattr(owner, name, explode)

        import importlib
        try:
            reloaded = importlib.reload(_app_module())
            booted_app = reloaded.app
        except Exception as exc:                       # noqa: BLE001 — this IS the assertion
            failures.append(
                f"The Flask app failed to boot when {name}() raised: {exc!r}. "
                "A bug in the new startup logic took the entire AI service down."
            )
            booted_app = None
    else:
        booted_app = _app_module().app

    # ── Steps 3-4: the service is up and fully functional ───────────────────
    if booted_app is not None:
        booted_app.config['TESTING'] = True
        with booted_app.test_client() as client:
            resp = client.get('/api/ai/health')
            check(
                resp.status_code == 200,
                f"GET /api/ai/health returned {resp.status_code}, expected 200 — "
                "the service is not serving after a reconciliation failure",
            )
            if resp.status_code == 200:
                body = resp.get_json()
                check(
                    body.get('service') == 'SLT AI Module',
                    f"Health payload does not identify the service: {body}",
                )
                check(
                    'models' in body,
                    "Health payload carries no model status — the service booted "
                    "but is not fully functional",
                )

    assert not failures, "\n".join(f"  - {f}" for f in failures)


def test_health_endpoint_is_reachable_at_baseline():
    """
    Control for the test above: with nothing patched, the app boots and
    /api/ai/health answers 200. If this ever fails, AI-028's red is an
    environment problem rather than the missing-hook finding it reports.
    """
    app_mod = _app_module()
    app_mod.app.config['TESTING'] = True
    with app_mod.app.test_client() as client:
        resp = client.get('/api/ai/health')
        assert resp.status_code == 200
        assert resp.get_json().get('service') == 'SLT AI Module'
