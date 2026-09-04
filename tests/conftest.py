"""
tests/conftest.py — Shared pytest Fixtures
==========================================
Provides fixtures used across all three test modules:
    - test_forecasting.py
    - test_clustering.py
    - test_routes.py

Key responsibilities:
    1. Patch is_db_available() → False so every test runs in
       synthetic-data mode with no real MySQL connection required.
    2. Build and yield the Flask test client once per session
       (expensive model initialisation happens only once).
    3. Provide ready-made DataFrames for unit tests.
    4. Ensure models/saved/ and logs/ directories exist before
       any test touches them.

pytest auto-discovers this file; no import needed in test files.
"""

import sys
import os
import json
import pathlib
import logging
import pytest

# ── Make project root importable ─────────────────────────────────────────────
# Needed when pytest is run from the project root OR from tests/
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ── Silence noisy loggers during tests ───────────────────────────────────────
logging.getLogger('cmdstanpy').setLevel(logging.CRITICAL)
logging.getLogger('prophet').setLevel(logging.CRITICAL)
logging.getLogger('numba').setLevel(logging.CRITICAL)
logging.getLogger('slt_ai').setLevel(logging.WARNING)


# ═════════════════════════════════════════════════════════════════════════════
# DIRECTORY SETUP
# ═════════════════════════════════════════════════════════════════════════════

def pytest_configure(config):
    """
    Called once before any test collection.
    Creates required directories so model persistence and logging
    never fail with FileNotFoundError during the test run.
    """
    for d in ['models/saved', 'logs']:
        pathlib.Path(PROJECT_ROOT / d).mkdir(parents=True, exist_ok=True)


# ═════════════════════════════════════════════════════════════════════════════
# DB MOCK — SESSION-SCOPED
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope='session', autouse=True)
def mock_db_unavailable(monkeypatch_session):
    """
    Patches config.is_db_available() to always return False for the
    entire test session.  This guarantees:
      - No real MySQL connection is attempted
      - All data paths use SyntheticDataGenerator
      - Tests pass on any developer machine or CI runner

    autouse=True means every test gets this fixture automatically —
    no explicit fixture request needed in test classes.
    """
    import config as cfg
    monkeypatch_session.setattr(cfg, 'is_db_available', lambda: False)
    yield


@pytest.fixture(scope='session')
def monkeypatch_session(request):
    """
    Session-scoped monkeypatch.
    pytest's built-in monkeypatch is function-scoped; this extends it
    to session scope so the DB patch persists across all tests.
    """
    from _pytest.monkeypatch import MonkeyPatch
    mp = MonkeyPatch()
    yield mp
    mp.undo()


# ═════════════════════════════════════════════════════════════════════════════
# FLASK TEST CLIENT — SESSION-SCOPED
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope='session')
def app():
    """
    Build and configure the Flask application once per session.
    Returns the configured Flask app (not a test client).

    Note: app.py defines its routes directly (not via the routes/
    Blueprint package — that package is an alternate, more granular
    route layer that isn't wired into the running service). Tests must
    exercise app.py's actual routes, so no blueprint registration
    happens here.
    """
    import app as app_module

    app_module.app.config.update({
        'TESTING':                 True,
        'PROPAGATE_EXCEPTIONS':    False,
        'JSON_SORT_KEYS':          False,
    })

    # Ensure saved-model and log directories exist
    pathlib.Path(PROJECT_ROOT / 'models' / 'saved').mkdir(parents=True, exist_ok=True)
    pathlib.Path(PROJECT_ROOT / 'logs').mkdir(parents=True, exist_ok=True)

    return app_module.app


@pytest.fixture(scope='session')
def client(app):
    """
    Flask test client — session-scoped so the expensive model
    initialisation (Prophet, KMeans, Classifier) only happens once.

    Usage in test classes:
        def test_something(self, client):
            resp = client.get('/api/ai/health/ping')
    """
    with app.test_client() as c:
        yield c


# ═════════════════════════════════════════════════════════════════════════════
# SYNTHETIC DATA FIXTURES — MODULE-SCOPED
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope='module')
def synth_gen():
    """
    SyntheticDataGenerator instance with fixed seed=42.
    Module-scoped: shared across all tests in the same test file.
    """
    from data.synthetic_data import SyntheticDataGenerator
    return SyntheticDataGenerator(seed=42)


@pytest.fixture(scope='module')
def raw_time_series(synth_gen):
    """
    Raw 365-day fault time-series DataFrame [ds, y].
    Used by forecasting tests.
    """
    return synth_gen.fault_time_series(days=365)


@pytest.fixture(scope='module')
def short_time_series(synth_gen):
    """
    Short 90-day time-series for quick tests.
    """
    return synth_gen.fault_time_series(days=90)


@pytest.fixture(scope='module')
def raw_gps_large(synth_gen):
    """
    Large GPS DataFrame (1000 points) for clustering tests.
    """
    return synth_gen.fault_gps_points(n=1000)


@pytest.fixture(scope='module')
def raw_gps_small(synth_gen):
    """
    Small GPS DataFrame (50 points) for edge-case tests.
    """
    return synth_gen.fault_gps_points(n=50)


@pytest.fixture(scope='module')
def technician_locations(synth_gen):
    """
    25 synthetic technician location records.
    Used by route optimisation tests.
    """
    return synth_gen.technician_locations(n=25)


@pytest.fixture(scope='module')
def technician_locations_small(synth_gen):
    """
    5 synthetic technician locations for lightweight graph tests.
    """
    return synth_gen.technician_locations(n=5)


# ═════════════════════════════════════════════════════════════════════════════
# CLEANED DATA FIXTURES — MODULE-SCOPED
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope='module')
def clean_time_series(raw_time_series):
    """
    Cleaned and validated time-series DataFrame ready for Prophet.
    Returns (DataFrame, stats_dict).
    Skips the test module if cleaning produces no data.
    """
    from data.data_cleaner import DataCleaner
    cleaner = DataCleaner()
    clean_df, stats = cleaner.clean_time_series(raw_time_series, min_rows=30)
    if clean_df is None:
        pytest.skip(f"Time-series cleaning produced no data: {stats.get('issues')}")
    return clean_df, stats


@pytest.fixture(scope='module')
def enriched_time_series(clean_time_series):
    """
    Cleaned + regressor-enriched DataFrame.
    Adds is_holiday, is_weekend, monsoon flags, rolling means, lags.
    """
    from data.feature_engineer import FeatureEngineer
    clean_df, _ = clean_time_series
    fe = FeatureEngineer()
    return fe.add_prophet_regressors(clean_df)


@pytest.fixture(scope='module')
def clean_gps(raw_gps_large):
    """
    Cleaned GPS DataFrame (latitude, longitude within SL bounds).
    Returns (DataFrame, stats_dict).
    """
    from data.data_cleaner import DataCleaner
    cleaner = DataCleaner()
    clean_df, stats = cleaner.clean_gps_points(raw_gps_large, min_points=20)
    if clean_df is None:
        pytest.skip(f"GPS cleaning produced no data: {stats.get('issues')}")
    return clean_df, stats


@pytest.fixture(scope='module')
def clean_technician_locs(technician_locations):
    """
    Cleaned technician location DataFrame.
    Removes any rows with null/out-of-bounds GPS.
    """
    from data.data_cleaner import DataCleaner
    result = DataCleaner().clean_technician_locations(technician_locations)
    if result is None or result.empty:
        pytest.skip("Technician location cleaning produced no data")
    return result


# ═════════════════════════════════════════════════════════════════════════════
# MODEL FIXTURES — SESSION-SCOPED (expensive, create once)
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope='session')
def forecaster():
    """
    ProphetForecaster singleton.
    Session-scoped: Prophet model is trained once and reused.
    """
    from models.forecasting import ProphetForecaster
    return ProphetForecaster()


@pytest.fixture(scope='session')
def clusterer():
    """
    KMeansClusterer singleton.
    Session-scoped: K-Means model is fitted once and reused.
    """
    from models.clustering import KMeansClusterer
    return KMeansClusterer()


@pytest.fixture(scope='session')
def classifier():
    """
    FaultClassifier singleton.
    Session-scoped: TF-IDF + LogReg trained once (auto-trains from synthetic).
    """
    from models.classifier import FaultClassifier
    return FaultClassifier()


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def json_response():
    """
    Helper fixture that decodes a Flask test response to a dict.

    Usage:
        def test_something(self, client, json_response):
            body = json_response(client.get('/api/ai/health/ping'))
            assert body['status'] == 'ok'
    """
    def _decode(response):
        return json.loads(response.data.decode('utf-8'))
    return _decode


@pytest.fixture
def assert_envelope():
    """
    Helper fixture that asserts standard API response envelope.

    Usage:
        def test_success(self, client, assert_envelope, json_response):
            body = json_response(client.get('/api/ai/health'))
            assert_envelope(body, success=True)
    """
    def _check(body: dict, success: bool = True):
        assert isinstance(body, dict), f"Response is not a dict: {type(body)}"
        assert body.get('success') is success, \
            f"Expected success={success}, got: {body.get('success')}"
        assert 'timestamp' in body, "Missing 'timestamp' in response"
        if success:
            assert 'data'    in body, "Missing 'data' in success response"
            assert 'message' in body, "Missing 'message' in success response"
        else:
            assert 'error' in body, "Missing 'error' in error response"
    return _check


# ── Colombo coordinates used by multiple route tests ──────────────────────────
@pytest.fixture
def colombo_coords():
    """Standard Colombo test coordinates."""
    return {'lat': 6.9271, 'lng': 79.8612}


@pytest.fixture
def kandy_coords():
    """Standard Kandy test coordinates."""
    return {'lat': 7.2906, 'lng': 80.6337}


@pytest.fixture
def out_of_bounds_coords():
    """Coordinates outside Sri Lanka (London) — should trigger 400."""
    return {'lat': 51.5074, 'lng': -0.1278}
