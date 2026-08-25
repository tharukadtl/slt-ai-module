"""
tests/test_routes.py — Flask Route Integration Tests
=====================================================
Tests the AI module's actual HTTP surface as defined directly in app.py
(health, predictions, clusters, dashboard, the async training endpoints).
All tests run without a real MySQL database (uses synthetic data — see
conftest.py's session-scoped mock_db_unavailable fixture).

Note: routes/*.py is a separate, more granular Blueprint-based route
layer that is NOT registered by app.py in production (app.py defines
its own routes directly). It is intentionally not exercised here —
testing it against app.py's URL space previously caused duplicate,
colliding route registrations and 8 silently-wrong test failures.

Covers:
  - Health endpoint
  - Predictions endpoint (forecast, horizon clamping)
  - Clusters endpoint (n_clusters clamping)
  - Route optimisation (POST body, validation, technicians-in-body)
  - Dashboard (combined widget payload)
  - POST /api/ai/train + GET /api/ai/train/status/{jobId} (SRS 5.6.7 async
    training contract — includes upload validation, since that's now
    the only path a CSV is ever submitted through). Replaces the old
    synchronous POST /api/ai/retrain (zero frontend callers, removed in
    Stage C) and POST /api/ai/upload-training-data + its /status +
    source=upload (all removed once frontend-admin's Model Training page
    migrated to this endpoint in Stage D — confirmed no remaining
    callers anywhere in frontend-admin/src before removal)
  - Input validation (bad params, missing required, out-of-bounds)
  - HTTP status codes and response envelope structure

Run:
    cd slt-ai-module
    python -m pytest tests/test_routes.py -v
    python -m pytest tests/test_routes.py -v -k "upload"   # filter
"""

import io
import sys
import os
import json
import math
import time
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ═════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope='module')
def client():
    """
    Flask test client for the real app.py application, with DB connectivity
    disabled (synthetic data mode) so tests never need a real MySQL server.
    """
    import config as cfg_mod
    original = cfg_mod.is_db_available
    cfg_mod.is_db_available = lambda: False

    import app as app_mod
    app_mod.app.config['TESTING'] = True
    with app_mod.app.test_client() as c:
        yield c

    cfg_mod.is_db_available = original


@pytest.fixture(scope='module')
def classifier():
    """
    FaultClassifier instance (auto-trains from synthetic data on construction).
    Module-scoped so the TF-IDF + LogReg pipelines are only built once.
    Used by TestFaultClassifier (sheet row AI-013).
    """
    from models.classifier import FaultClassifier
    return FaultClassifier()


def json_body(response) -> dict:
    """Parse JSON from a Flask test response."""
    return json.loads(response.data.decode('utf-8'))


def independent_haversine_km(lat1, lng1, lat2, lng2) -> float:
    """
    Great-circle distance in km, computed from scratch here in the test so the
    assertion on distanceKm does not simply re-use the production haversine_km()
    implementation it is meant to verify. Standard Haversine, R=6371 km.
    """
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi       = math.radians(lat2 - lat1)
    dlambda    = math.radians(lng2 - lng1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ─── Shared assertion helpers ─────────────────────────────────────────────────

def assert_success_envelope(body: dict) -> None:
    """Assert standard success response shape."""
    assert body.get('success') is True, f"Expected success=True, got: {body}"
    assert 'data'      in body
    assert 'message'   in body
    assert 'timestamp' in body


def assert_error_envelope(body: dict) -> None:
    """Assert standard error response shape."""
    assert body.get('success') is False
    assert 'error'     in body
    assert 'timestamp' in body


# ═════════════════════════════════════════════════════════════════════════════
# HEALTH ENDPOINT TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestHealthEndpoint:

    def test_health_returns_200(self, client):
        resp = client.get('/api/ai/health')
        assert resp.status_code == 200

    def test_health_has_service_key(self, client):
        body = json_body(client.get('/api/ai/health'))
        assert body.get('service') == 'SLT AI Module'

    def test_health_has_models(self, client):
        body = json_body(client.get('/api/ai/health'))
        assert 'models' in body
        for key in ('prophet_forecaster', 'kmeans_clusterer', 'dijkstra_router'):
            assert key in body['models']

    def test_health_has_config(self, client):
        body = json_body(client.get('/api/ai/health'))
        assert 'config' in body

    def test_health_db_status_shows_unavailable(self, client):
        """In test mode (DB mocked off), db_status should reflect synthetic fallback."""
        body = json_body(client.get('/api/ai/health'))
        assert 'unavailable' in body.get('db_status', '')


# ═════════════════════════════════════════════════════════════════════════════
# PREDICTIONS ENDPOINT TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestPredictionsEndpoint:

    def test_predictions_default_returns_200(self, client):
        resp = client.get('/api/ai/predictions')
        assert resp.status_code == 200

    def test_predictions_returns_success_envelope(self, client):
        body = json_body(client.get('/api/ai/predictions'))
        assert_success_envelope(body)

    def test_predictions_data_has_required_keys(self, client):
        data = json_body(client.get('/api/ai/predictions'))['data']
        for key in ['historical', 'forecast', 'metrics', 'trend', 'nextPeriodTotal']:
            assert key in data, f"Missing data key: {key}"

    def test_predictions_7_day_horizon(self, client):
        resp = client.get('/api/ai/predictions?horizon=7')
        assert resp.status_code == 200
        data = json_body(resp)['data']
        assert len(data['forecast']) == 7

    def test_predictions_30_day_horizon(self, client):
        data = json_body(client.get('/api/ai/predictions?horizon=30'))['data']
        assert len(data['forecast']) == 30

    def test_predictions_horizon_capped_at_90(self, client):
        """horizon=999 should be clamped to 90."""
        resp = client.get('/api/ai/predictions?horizon=999')
        assert resp.status_code == 200
        data = json_body(resp)['data']
        assert len(data['forecast']) == 90

    def test_predictions_horizon_floored_at_7(self, client):
        """horizon=1 should be clamped to 7."""
        resp = client.get('/api/ai/predictions?horizon=1')
        assert resp.status_code == 200
        data = json_body(resp)['data']
        assert len(data['forecast']) == 7

    def test_predictions_data_source_is_synthetic(self, client):
        data = json_body(client.get('/api/ai/predictions'))['data']
        assert data.get('dataSource') == 'synthetic'

    def test_predictions_forecast_yhat_non_negative(self, client):
        data = json_body(client.get('/api/ai/predictions?horizon=7'))['data']
        for item in data['forecast']:
            assert item['yhat'] >= 0

    def test_predictions_forecast_has_upper_lower(self, client):
        data = json_body(client.get('/api/ai/predictions?horizon=7'))['data']
        for item in data['forecast']:
            assert 'upper' in item and 'lower' in item

    def test_predictions_next_total_int(self, client):
        data = json_body(client.get('/api/ai/predictions'))['data']
        assert isinstance(data['nextPeriodTotal'], int)

    def test_predictions_trend_valid(self, client):
        data = json_body(client.get('/api/ai/predictions'))['data']
        assert data['trend'] in ('up', 'down', 'stable')


# ═════════════════════════════════════════════════════════════════════════════
# CLUSTERS ENDPOINT TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestClustersEndpoint:

    def test_clusters_default_returns_200(self, client):
        resp = client.get('/api/ai/clusters')
        assert resp.status_code == 200

    def test_clusters_returns_success_envelope(self, client):
        body = json_body(client.get('/api/ai/clusters'))
        assert_success_envelope(body)

    def test_clusters_data_has_required_keys(self, client):
        data = json_body(client.get('/api/ai/clusters'))['data']
        for key in ['clusters', 'totalFaults', 'nClusters']:
            assert key in data, f"Missing key: {key}"

    def test_clusters_default_k5(self, client):
        data = json_body(client.get('/api/ai/clusters'))['data']
        assert len(data['clusters']) == 5

    def test_clusters_custom_k3(self, client):
        data = json_body(client.get('/api/ai/clusters?n_clusters=3'))['data']
        assert len(data['clusters']) == 3

    def test_clusters_k_clamped_to_2_minimum(self, client):
        data = json_body(client.get('/api/ai/clusters?n_clusters=1'))['data']
        assert len(data['clusters']) >= 2

    def test_clusters_data_source_synthetic(self, client):
        data = json_body(client.get('/api/ai/clusters'))['data']
        assert data.get('dataSource') == 'synthetic'

    def test_clusters_fault_counts_non_negative(self, client):
        data = json_body(client.get('/api/ai/clusters'))['data']
        for c in data['clusters']:
            assert c['faultCount'] >= 0

    def test_clusters_risk_levels_valid(self, client):
        data = json_body(client.get('/api/ai/clusters'))['data']
        for c in data['clusters']:
            assert c['riskLevel'] in ('HIGH', 'MEDIUM', 'LOW')

    def test_clusters_have_centroid(self, client):
        data = json_body(client.get('/api/ai/clusters'))['data']
        for c in data['clusters']:
            assert 'lat' in c['centroid'] and 'lng' in c['centroid']

    # Note: see the equivalent comment in TestPredictionsEndpoint — the
    # "no CSV uploaded yet" precondition isn't reliably reproducible here
    # since UploadManager persists the last upload to disk across runs.


# ═════════════════════════════════════════════════════════════════════════════
# DASHBOARD ENDPOINT TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestDashboardEndpoint:

    def test_dashboard_returns_200(self, client):
        resp = client.get('/api/ai/dashboard')
        assert resp.status_code == 200

    def test_dashboard_success_envelope(self, client):
        body = json_body(client.get('/api/ai/dashboard'))
        assert_success_envelope(body)

    def test_dashboard_has_required_sections(self, client):
        data = json_body(client.get('/api/ai/dashboard'))['data']
        for key in ['modelsReady', 'dbAvailable', 'generatedAt']:
            assert key in data, f"Missing dashboard section: {key}"

    def test_dashboard_db_available_is_false_in_test_mode(self, client):
        data = json_body(client.get('/api/ai/dashboard'))['data']
        assert data['dbAvailable'] is False

    def test_dashboard_models_ready_flags(self, client):
        data = json_body(client.get('/api/ai/dashboard'))['data']
        for key in ('forecaster', 'clusterer', 'router'):
            assert key in data['modelsReady']


# ═════════════════════════════════════════════════════════════════════════════
# SHORTEST-PATH ENDPOINT TESTS (FR-29 / SRS 5.6.6 — POST /api/ai/shortest-path)
# ═════════════════════════════════════════════════════════════════════════════

class TestShortestPathEndpoint:
    """
    Verifies the Technician-to-fault Dijkstra shortest-path endpoint
    (Stage 1: AI/ML backend contract only). Every request is currently
    served via the SRS-sanctioned Haversine straight-line fallback
    (routed=False) because no road-network graph data source exists.

    Valid SL coords: Colombo 6.9271/79.8612, Kandy 7.2906/80.6337.
    SL bounds: lat 5.9–9.9, lng 79.5–81.9.
    """

    CURRENT = {'lat': 6.9271, 'lng': 79.8612}   # Colombo (technician)
    FAULT   = {'lat': 7.2906, 'lng': 80.6337}   # Kandy (fault)

    def _valid_body(self) -> dict:
        return {
            'currentLat': self.CURRENT['lat'], 'currentLng': self.CURRENT['lng'],
            'faultLat':   self.FAULT['lat'],   'faultLng':   self.FAULT['lng'],
        }

    def _post(self, client, body):
        return client.post('/api/ai/shortest-path', json=body)

    # ─── 1. Valid coordinates → 200 + correct payload ────────────────────────

    def test_valid_returns_200_success_envelope(self, client):
        resp = self._post(client, self._valid_body())
        assert resp.status_code == 200
        body = json_body(resp)
        assert_success_envelope(body)

    def test_valid_waypoints_are_exactly_start_and_end(self, client):
        data = json_body(self._post(client, self._valid_body()))['data']
        wps = data['waypoints']
        assert isinstance(wps, list)
        assert len(wps) == 2, f"Expected 2 waypoints today, got {len(wps)}: {wps}"
        # First waypoint == technician current location, second == fault location
        assert wps[0]['lat'] == pytest.approx(self.CURRENT['lat'])
        assert wps[0]['lng'] == pytest.approx(self.CURRENT['lng'])
        assert wps[1]['lat'] == pytest.approx(self.FAULT['lat'])
        assert wps[1]['lng'] == pytest.approx(self.FAULT['lng'])

    def test_valid_distance_matches_independent_haversine(self, client):
        data = json_body(self._post(client, self._valid_body()))['data']
        expected = independent_haversine_km(
            self.CURRENT['lat'], self.CURRENT['lng'],
            self.FAULT['lat'], self.FAULT['lng'],
        )
        assert data['distanceKm'] > 0
        # Colombo→Kandy great-circle is ~94 km; assert we're in the right ballpark
        assert 80 < data['distanceKm'] < 110, f"distanceKm implausible: {data['distanceKm']}"
        # And that it matches an independently-computed great-circle distance
        # (response value is rounded to 3 dp in the router).
        assert data['distanceKm'] == pytest.approx(expected, abs=1e-2)

    def test_valid_eta_consistent_with_distance_and_speed(self, client):
        data = json_body(self._post(client, self._valid_body()))['data']
        assert isinstance(data['etaMinutes'], int)
        assert data['etaMinutes'] > 0
        speed = data['avgSpeedKmh']
        assert speed > 0
        expected_eta = (data['distanceKm'] / speed) * 60
        # etaMinutes is round()-ed by the router from the (unrounded) distance,
        # so allow ±1 minute vs recomputing from the 3-dp-rounded distanceKm.
        assert abs(data['etaMinutes'] - expected_eta) <= 1, \
            f"etaMinutes {data['etaMinutes']} inconsistent with " \
            f"distanceKm {data['distanceKm']} / avgSpeedKmh {speed}"

    def test_valid_avg_speed_matches_config(self, client):
        from config import Config
        data = json_body(self._post(client, self._valid_body()))['data']
        assert data['avgSpeedKmh'] == pytest.approx(Config.ROUTE_AVG_SPEED_KMH)

    # ─── 2. Out-of-bounds coordinates → 400 ──────────────────────────────────

    def test_out_of_bounds_current_lat_returns_400(self, client):
        body = self._valid_body()
        body['currentLat'] = 60.0   # far north of Sri Lanka
        resp = self._post(client, body)
        assert resp.status_code == 400
        assert_error_envelope(json_body(resp))

    def test_out_of_bounds_fault_lng_returns_400(self, client):
        body = self._valid_body()
        body['faultLng'] = 200.0    # nonsensical longitude
        resp = self._post(client, body)
        assert resp.status_code == 400
        assert_error_envelope(json_body(resp))

    def test_out_of_bounds_error_names_the_offending_endpoint(self, client):
        """Error should distinguish technician-location vs fault-location failures."""
        body = self._valid_body()
        body['faultLat'] = 60.0
        err_msg = json_body(self._post(client, body))['error']
        assert 'Fault location' in err_msg, f"Unexpected error text: {err_msg}"

    # ─── 3. Missing required fields → 400 ────────────────────────────────────

    def test_missing_fault_lat_returns_400(self, client):
        body = self._valid_body()
        del body['faultLat']
        resp = self._post(client, body)
        assert resp.status_code == 400
        assert_error_envelope(json_body(resp))

    def test_empty_object_returns_400(self, client):
        resp = self._post(client, {})
        assert resp.status_code == 400
        assert_error_envelope(json_body(resp))

    def test_no_body_at_all_returns_400(self, client):
        resp = client.post('/api/ai/shortest-path')
        assert resp.status_code == 400
        assert_error_envelope(json_body(resp))

    # ─── 4. Malformed input → 400 (never a 500 crash) ────────────────────────

    def test_non_numeric_coords_returns_400_not_500(self, client):
        body = {
            'currentLat': 'abc', 'currentLng': 'xyz',
            'faultLat': 'def', 'faultLng': 'ghi',
        }
        resp = self._post(client, body)
        assert resp.status_code == 400, \
            f"Expected 400 for non-numeric coords, got {resp.status_code}"
        assert_error_envelope(json_body(resp))

    def test_array_body_returns_400_not_500(self, client):
        """A JSON array instead of an object must be rejected as 400, not crash."""
        resp = client.post(
            '/api/ai/shortest-path',
            data=json.dumps([1, 2, 3]),
            content_type='application/json',
        )
        assert resp.status_code == 400, \
            f"Expected 400 for array body, got {resp.status_code}"
        assert_error_envelope(json_body(resp))

    def test_null_coord_value_returns_400_not_500(self, client):
        body = self._valid_body()
        body['currentLat'] = None
        resp = self._post(client, body)
        assert resp.status_code == 400
        assert_error_envelope(json_body(resp))

    # ─── 5. routed=False fallback semantics ──────────────────────────────────

    def test_routed_is_explicitly_false(self, client):
        data = json_body(self._post(client, self._valid_body()))['data']
        assert data['routed'] is False, \
            "No road-network data source exists today; routed must be False"

    def test_fallback_is_signalled_to_client(self, client):
        """
        A mobile client must not be able to mistake this for a real routed
        polyline: both the algorithm field and the message make the
        Haversine straight-line fallback nature explicit.
        """
        body = json_body(self._post(client, self._valid_body()))
        data = body['data']
        assert 'haversine' in data['algorithm'].lower()
        assert ('fallback' in data['algorithm'].lower()
                or 'straight-line' in data['algorithm'].lower())
        # And the human-readable envelope message also flags the fallback.
        assert 'fallback' in body['message'].lower()

    # ─── 6. Response-shape future-proofing ───────────────────────────────────

    def test_response_shape_is_stable_and_typed(self, client):
        """
        The top-level keys must be stable and typed such that a future real
        routing engine can populate `waypoints` with many intermediate points
        and flip `routed` to True without any key changing shape/type.
        """
        data = json_body(self._post(client, self._valid_body()))['data']
        for key in ('waypoints', 'distanceKm', 'etaMinutes',
                    'routed', 'algorithm', 'avgSpeedKmh'):
            assert key in data, f"Missing stable top-level key: {key}"

        assert isinstance(data['waypoints'], list)
        # waypoints is a list of {lat, lng} dicts regardless of length —
        # a real multi-waypoint route would just be a longer list of the same.
        for wp in data['waypoints']:
            assert isinstance(wp, dict)
            assert set(wp.keys()) == {'lat', 'lng'}, f"Unexpected waypoint keys: {wp}"
            assert isinstance(wp['lat'], (int, float))
            assert isinstance(wp['lng'], (int, float))

        assert isinstance(data['distanceKm'], (int, float))
        assert isinstance(data['etaMinutes'], int)
        assert isinstance(data['routed'], bool)
        assert isinstance(data['algorithm'], str)
        assert isinstance(data['avgSpeedKmh'], (int, float))

    # ─── 7. Wrong HTTP method → 405 ──────────────────────────────────────────

    def test_get_method_returns_405(self, client):
        """shortest-path is POST-only; GET should return 405."""
        resp = client.get('/api/ai/shortest-path')
        assert resp.status_code == 405


# ═════════════════════════════════════════════════════════════════════════════
# ASYNC TRAINING ENDPOINT TESTS (SRS 5.6.7 — POST /api/ai/train + status poll)
# ═════════════════════════════════════════════════════════════════════════════

class TestTrainEndpoint:

    VALID_CSV = (
        b"date,latitude,longitude,category\n"
        b"2026-01-01,6.9271,79.8612,BROADBAND\n"
        b"2026-01-02,7.2906,80.6337,FIBER\n"
        b"2026-01-03,6.0535,80.2210,TELEPHONE\n"
        b"2026-01-04,9.6615,80.0255,TELEVISION\n"
    )

    def test_train_no_file_returns_400(self, client):
        resp = client.post('/api/ai/train', data={}, content_type='multipart/form-data')
        assert resp.status_code == 400

    def test_train_non_csv_returns_400(self, client):
        resp = client.post(
            '/api/ai/train',
            data={'file': (io.BytesIO(b'not a csv'), 'data.txt')},
            content_type='multipart/form-data'
        )
        assert resp.status_code == 400

    def test_train_returns_202_with_job_id_and_upload_summary(self, client):
        resp = client.post(
            '/api/ai/train',
            data={'file': (io.BytesIO(self.VALID_CSV), 'faults.csv')},
            content_type='multipart/form-data'
        )
        assert resp.status_code == 202
        data = json_body(resp)['data']
        assert 'jobId' in data
        assert data['status'] == 'queued'
        summary = data['uploadSummary']
        assert summary['rowCount'] == 4
        assert summary['hasGpsColumns'] is True
        assert summary['usableForForecasting'] is True
        assert summary['usableForClustering'] is True

    def test_train_status_unknown_job_returns_404(self, client):
        resp = client.get('/api/ai/train/status/does-not-exist')
        assert resp.status_code == 404

    def test_train_job_reaches_complete(self, client):
        """
        4 rows is enough to exercise the full async lifecycle even though
        it's below both models' minimum training thresholds — each model's
        result should show status='error' ("insufficient data"), and the
        job itself should still reach 'complete' (a per-model training
        failure is not a job failure).
        """
        resp = client.post(
            '/api/ai/train',
            data={'file': (io.BytesIO(self.VALID_CSV), 'faults.csv')},
            content_type='multipart/form-data'
        )
        job_id = json_body(resp)['data']['jobId']

        job = None
        deadline = time.time() + 15
        while time.time() < deadline:
            status_resp = client.get(f'/api/ai/train/status/{job_id}')
            assert status_resp.status_code == 200
            job = json_body(status_resp)['data']
            if job['status'] in ('complete', 'failed'):
                break
            time.sleep(0.2)

        assert job is not None
        assert job['status'] == 'complete', f"Job did not complete in time: {job}"
        assert 'forecaster' in job['result']
        assert 'clusterer' in job['result']

    # ─── AI-021 (sheet 10_AI_MODULE, FR-30) ──────────────────────────────────

    @staticmethod
    def _trainable_csv(days: int = 220, per_day: int = 2) -> bytes:
        """
        A CSV that is genuinely large enough to train BOTH models, unlike
        VALID_CSV above (4 rows, deliberately below both thresholds):
          - forecaster: retrain() cleans with min_rows=Config.FORECAST_MIN_HISTORY_DAYS
            (180), so >= 180 distinct dates are needed once the rows are grouped
            into daily counts — 220 gives headroom above that minimum, same
            margin as test_forecasting.py's raw_ts fixture.
          - clusterer:  app.py's _run_training_job requires >= 10 GPS rows.
        Every date sits inside the SRS 5.6.7 24-month window (counted back from
        today) and every coordinate inside the Sri Lanka bounds, so nothing is
        excluded by UploadManager's validation. Coordinates are jittered around
        four real cities rather than repeated verbatim, because DataCleaner's
        GPS pass de-duplicates identical points — four distinct coordinates
        repeated 30x each collapse to 4 usable points and fail the >= 10 check.
        """
        from datetime import date, timedelta
        # Four real Sri Lankan cities so the GPS points form genuine clusters.
        cities = [(6.9271, 79.8612), (7.2906, 80.6337),
                  (6.0535, 80.2210), (9.6615, 80.0255)]
        cats   = ['BROADBAND', 'FIBER', 'TELEPHONE', 'TELEVISION']
        lines  = ['date,latitude,longitude,category']
        today  = date.today()
        for d in range(days):
            day = today - timedelta(days=days - d)
            for n in range(per_day):
                idx = (d + n) % len(cities)
                lat, lng = cities[idx]
                jitter = ((d * per_day) + n) * 0.0007      # ~75 m steps, stays in-bounds
                lines.append(
                    f"{day.isoformat()},{lat + jitter:.6f},{lng + jitter:.6f},"
                    f"{cats[(d + n) % len(cats)]}"
                )
        return ('\n'.join(lines) + '\n').encode()

    def test_async_job_lifecycle(self, client):
        """
        AI-021 — POST /api/ai/train is genuinely asynchronous (202 + jobId,
        returned without waiting for training), the job is pollable through to
        completion, and the finished job carries a candidate + comparison for
        BOTH models rather than only reaching a terminal state.

        The elapsed-time check is what distinguishes "async" from "fast": a
        synchronous implementation would have to fit Prophet (multiple seconds)
        before it could answer, so a sub-2s 202 could not happen by accident.
        """
        csv_bytes = self._trainable_csv()

        started = time.time()
        resp = client.post(
            '/api/ai/train',
            data={'file': (io.BytesIO(csv_bytes), 'faults.csv')},
            content_type='multipart/form-data'
        )
        post_seconds = time.time() - started

        assert resp.status_code == 202, \
            f"Expected 202 Accepted, got {resp.status_code}: {resp.data[:400]}"
        data = json_body(resp)['data']
        job_id = data['jobId']
        assert job_id
        assert data['status'] == 'queued'
        assert post_seconds < 2.0, (
            f"POST /api/ai/train took {post_seconds:.1f}s — it appears to be "
            "blocking on training rather than returning a jobId immediately"
        )

        # Every row is in-window and well-formed, so nothing should be skipped.
        summary = data['uploadSummary']
        assert summary['rowCount'] == 440
        assert summary['skippedRows']['tooOld'] == 0
        assert summary['skippedRows']['badDates'] == 0
        assert summary['usableForForecasting'] is True
        assert summary['usableForClustering'] is True

        # Poll to a terminal state, recording that a non-terminal status was
        # actually observable (i.e. the job really did run in the background).
        seen = set()
        job = None
        deadline = time.time() + 180
        while time.time() < deadline:
            status_resp = client.get(f'/api/ai/train/status/{job_id}')
            assert status_resp.status_code == 200
            job = json_body(status_resp)['data']
            seen.add(job['status'])
            if job['status'] in ('complete', 'failed'):
                break
            time.sleep(0.25)

        assert job is not None
        assert job['status'] == 'complete', \
            f"Job did not complete (statuses seen: {sorted(seen)}): {job}"

        for model_key in ('forecaster', 'clusterer'):
            entry = job['result'].get(model_key)
            assert entry is not None, f"No '{model_key}' entry in the job result"
            assert entry.get('status') == 'candidate', (
                f"{model_key} did not produce a candidate: {entry}"
            )
            assert entry.get('versionId') is not None
            assert 'comparison' in entry, \
                f"{model_key} candidate carries no comparison against the active version"

    # ─── H3 — EXCHANGEAREA/OPMCCODE-only upload (no GPS at all) ───────────────
    # The real WFMS export shape, confirmed against a real sample: DATE +
    # EXCHANGEAREA + OPMCCODE columns, never latitude/longitude.

    @staticmethod
    def _exchange_area_only_csv(days: int = 220, per_day: int = 1) -> bytes:
        """
        date-only + EXCHANGEAREA/OPMCCODE, no GPS — enough distinct dates
        (>= Config.FORECAST_MIN_HISTORY_DAYS, 180, with headroom) for the
        forecaster to train, zero GPS rows for the clusterer so the
        categorical fallback is the only path available to it.
        """
        from datetime import date, timedelta
        areas = ['DGD', 'AD', 'KY', 'CEN', 'MHG', 'KG']
        opmcs = ['KTOP', 'ADOP', 'KYOP', 'MDOP', 'HOOP', 'KUOP']
        lines = ['DATE,EXCHANGEAREA,OPMCCODE,CATEGORY']
        today = date.today()
        for d in range(days):
            day = today - timedelta(days=days - d)
            for n in range(per_day):
                idx = (d + n) % len(areas)
                lines.append(
                    f"{day.isoformat()},{areas[idx]},{opmcs[idx]},BROADBAND"
                )
        return ('\n'.join(lines) + '\n').encode()

    def test_async_job_lifecycle_exchange_area_only(self, client):
        """
        H3 — a CSV with no GPS at all (the real WFMS shape) must still reach a
        completed job, with the forecaster training normally off `date` alone
        and the clusterer falling back to categorical EXCHANGEAREA grouping
        instead of erroring out with "no usable latitude/longitude columns."
        """
        csv_bytes = self._exchange_area_only_csv()

        resp = client.post(
            '/api/ai/train',
            data={'file': (io.BytesIO(csv_bytes), 'faults.csv')},
            content_type='multipart/form-data'
        )
        assert resp.status_code == 202, f"Body: {resp.data[:400]}"
        data = json_body(resp)['data']
        job_id = data['jobId']

        summary = data['uploadSummary']
        assert summary['hasGpsColumns'] is False
        assert summary['hasExchangeAreaColumn'] is True
        assert summary['hasOpmcCodeColumn'] is True
        assert summary['usableForClustering'] is True, (
            "A GPS-less upload with EXCHANGEAREA must still be usableForClustering "
            "via the categorical fallback"
        )
        assert summary['clusteringMethod'] == 'categorical'

        job = None
        deadline = time.time() + 180
        while time.time() < deadline:
            status_resp = client.get(f'/api/ai/train/status/{job_id}')
            assert status_resp.status_code == 200
            job = json_body(status_resp)['data']
            if job['status'] in ('complete', 'failed'):
                break
            time.sleep(0.25)

        assert job is not None
        assert job['status'] == 'complete', f"Job did not complete in time: {job}"

        forecaster_entry = job['result'].get('forecaster')
        assert forecaster_entry is not None
        assert forecaster_entry.get('status') == 'candidate', (
            f"Forecaster must still train off date-only data: {forecaster_entry}"
        )

        clusterer_entry = job['result'].get('clusterer')
        assert clusterer_entry is not None, "No 'clusterer' entry in the job result"
        assert clusterer_entry.get('status') == 'error', (
            "The categorical fallback has no fitted model/versionId to report as a "
            f"'candidate' — expected the documented 'error'-shaped entry, got: {clusterer_entry}"
        )
        assert 'EXCHANGEAREA' in clusterer_entry.get('message', ''), (
            f"The message must explain the categorical fallback was used: {clusterer_entry}"
        )

        preview = clusterer_entry.get('categoricalPreview')
        assert preview is not None, "categoricalPreview must carry the actual computed zones"
        assert preview['nClusters'] == 6, (
            f"Expected one zone per unique EXCHANGEAREA code (6), got {preview['nClusters']}"
        )
        assert preview['totalFaults'] == 220
        assert preview['silhouetteScore'] is None
        region_names = {c['regionName'] for c in preview['clusters']}
        assert region_names == {'DGD', 'AD', 'KY', 'CEN', 'MHG', 'KG'}
        for c in preview['clusters']:
            assert c['centroid'] is None
            assert c['density'] is None


# ═════════════════════════════════════════════════════════════════════════════
# MODEL VERSION GOVERNANCE — SHEET 10_AI_MODULE, ROW AI-024 (FR-30 / SRS 5.6.7)
# ═════════════════════════════════════════════════════════════════════════════

class TestModelVersions:
    """
    AI-024 — POST /api/ai/model-versions/<model>/activate and .../rollback.

    The consolidated QA report records these two endpoints as the one remaining
    coverage gap in the SRS 5.6.7 governance flow ("no permanent automated test
    covers the activate/rollback HTTP endpoints themselves"); this closes it.

    Both endpoints mutate the on-disk registry that the running service loads at
    boot, so the forecaster's registry is swapped for a throwaway one rooted in
    tmp_path for the duration of the test. `_model`/`_meta`/`_trained` are
    monkeypatched too (not just `_registry`) because activate_version() and
    rollback() write to all four, and monkeypatch restores every one of them at
    teardown — otherwise this test would leave the live singleton holding a
    stand-in object.
    """

    @pytest.fixture
    def registry(self, monkeypatch, tmp_path):
        import app as app_mod
        from models.model_registry import ModelVersionRegistry

        fc  = app_mod._forecaster
        reg = ModelVersionRegistry(tmp_path, 'forecaster')
        monkeypatch.setattr(fc, '_registry', reg, raising=False)
        monkeypatch.setattr(fc, '_model',    fc._model,    raising=False)
        monkeypatch.setattr(fc, '_meta',     fc._meta,     raising=False)
        monkeypatch.setattr(fc, '_trained',  fc._trained,  raising=False)

        active = reg.save_version(
            {'stand-in': 'v1 active'}, {'mae': 5.0, 'accuracy': 70.0},
            {'training_rows': 200}, status='active',
        )
        candidate = reg.save_version(
            {'stand-in': 'v2 candidate'}, {'mae': 3.0, 'accuracy': 88.0},
            {'training_rows': 400}, status='candidate',
        )
        return reg, active, candidate

    def _versions(self, client):
        resp = client.get('/api/ai/model-versions/forecaster')
        assert resp.status_code == 200
        return {v['versionId']: v['status']
                for v in json_body(resp)['data']['versions']}

    def test_activate_and_rollback(self, client, registry):
        reg, active, candidate = registry

        # ── 1. Pre-condition, read back through the real GET endpoint ─────────
        assert self._versions(client) == {
            active['versionId']: 'active', candidate['versionId']: 'candidate',
        }

        # ── 2. Activate the candidate ─────────────────────────────────────────
        resp = client.post('/api/ai/model-versions/forecaster/activate',
                            json={'versionId': candidate['versionId']})
        assert resp.status_code == 200, resp.data[:400]
        body = json_body(resp)
        assert_success_envelope(body)
        assert body['data']['versionId'] == candidate['versionId']
        assert body['data']['status'] == 'active'

        # New version active, previous one archived (not deleted, not candidate).
        assert self._versions(client) == {
            active['versionId']: 'archived', candidate['versionId']: 'active',
        }

        # ── 3. Roll back — the active pointer reverts to the previous version ─
        resp = client.post('/api/ai/model-versions/forecaster/rollback', json={})
        assert resp.status_code == 200, resp.data[:400]
        body = json_body(resp)
        assert_success_envelope(body)
        assert body['data']['versionId'] == active['versionId']

        assert self._versions(client) == {
            active['versionId']: 'active', candidate['versionId']: 'archived',
        }

        # ── 4. Invalid input is rejected cleanly, never a 500 ─────────────────
        resp = client.post('/api/ai/model-versions/forecaster/activate',
                            json={'versionId': 999999})
        assert resp.status_code == 404, \
            f"Nonexistent versionId should be a clean 404, got {resp.status_code}"
        assert_error_envelope(json_body(resp))

        resp = client.post('/api/ai/model-versions/forecaster/activate', json={})
        assert resp.status_code == 400
        assert_error_envelope(json_body(resp))

        resp = client.post('/api/ai/model-versions/not-a-model/activate',
                            json={'versionId': 1})
        assert resp.status_code == 400
        assert_error_envelope(json_body(resp))

        # The registry is unchanged by the three rejected calls.
        assert self._versions(client) == {
            active['versionId']: 'active', candidate['versionId']: 'archived',
        }


# ═════════════════════════════════════════════════════════════════════════════
# FAULT CLASSIFIER — SHEET 10_AI_MODULE, ROW AI-013 (FR-26)
# ═════════════════════════════════════════════════════════════════════════════

class TestFaultClassifier:
    """
    AI-013 — FaultClassifier.predict() category/priority behaviour.

    Mapped to this file by the sheet, though the classifier is a model rather
    than a route: app.py registers no classify endpoint at all (the
    GET/POST /api/ai/dashboard/classify handler lives in routes/dashboard.py,
    which is the unregistered Blueprint layer this file's header documents as
    intentionally not exercised). The model itself is therefore driven directly,
    which is the only way this row's assertions are reachable today.
    """

    def test_predict_broadband_category(self, classifier):
        failures = []

        def check(condition, message):
            if not condition:
                failures.append(message)

        # ── Step 1-2: a broadband description, with a usable confidence ───────
        result = classifier.predict('No internet since 08:00 router blinking red')
        check(result['category'] == 'BROADBAND',
              f"'No internet since 08:00 router blinking red' classified as "
              f"{result['category']!r}, expected 'BROADBAND'")
        check(0 < result['catConfidence'] <= 1.0,
              f"catConfidence {result['catConfidence']} outside (0, 1]")

        # ── Step 3: a fibre-cut description ──────────────────────────────────
        fiber = classifier.predict('Fiber cable cut total area outage')
        check(fiber['category'] == 'FIBER',
              f"'Fiber cable cut total area outage' classified as "
              f"{fiber['category']!r}, expected 'FIBER'")

        # ── Step 4: an emergency description escalates priority ──────────────
        hospital = classifier.predict('Hospital internet down urgent')
        check(hospital['priority'] == 'HIGH',
              f"'Hospital internet down urgent' got priority "
              f"{hospital['priority']!r}, expected 'HIGH'")
        check(hospital['urgencyScore'] >= 70,
              f"'Hospital internet down urgent' urgencyScore "
              f"{hospital['urgencyScore']} is below 70")

        assert not failures, "\n".join(f"  - {f}" for f in failures)


# ═════════════════════════════════════════════════════════════════════════════
# SYNTHETIC FALLBACK ACROSS ENDPOINTS — SHEET 10_AI_MODULE, ROW AI-014 (FR-26)
# ═════════════════════════════════════════════════════════════════════════════

class TestAllEndpoints:
    """
    AI-014 — with no database reachable, every AI endpoint must still answer
    successfully from synthetic data rather than erroring or hanging.

    The module `client` fixture already patches config.is_db_available to False
    for this whole file, which is exactly the row's "DB mocked unavailable"
    pre-condition; it is re-asserted here through /api/ai/health so the test
    proves the pre-condition rather than assuming the fixture applied.

    Endpoint-list substitution, stated rather than silently glossed: the row
    names GET /api/ai/optimize-route and POST /api/ai/dashboard/classify.
    Neither is registered by app.py — there is no optimize-route handler
    anywhere in the codebase, and classify exists only in the unregistered
    routes/dashboard.py Blueprint. The endpoints below are app.py's real,
    reachable data-serving surface instead.
    """

    def test_synthetic_fallback_when_db_down(self, client):
        # Pre-condition: the service itself agrees the DB is unavailable.
        health = json_body(client.get('/api/ai/health'))
        assert 'unavailable' in health.get('db_status', ''), \
            f"DB is not in the unavailable state this row requires: {health.get('db_status')}"

        failures = []

        # ── GET endpoints ────────────────────────────────────────────────────
        for path, expect_synthetic in [
            ('/api/ai/predictions?horizon=30', True),
            ('/api/ai/clusters?n_clusters=5',  True),
            ('/api/ai/resource-plan?horizon=7', False),
            ('/api/ai/dashboard',               False),
        ]:
            resp = client.get(path)
            if resp.status_code != 200:
                failures.append(f"GET {path} returned {resp.status_code}, expected 200")
                continue
            body = json_body(resp)
            if body.get('success') is not True:
                failures.append(f"GET {path} returned a non-success envelope: {body.get('error')}")
                continue
            if expect_synthetic and body['data'].get('dataSource') != 'synthetic':
                failures.append(
                    f"GET {path} reported dataSource="
                    f"{body['data'].get('dataSource')!r}, expected 'synthetic'"
                )

        # ── POST shortest-path — synthetic technicians are not needed, but the
        #     endpoint must still work with no DB behind it ────────────────────
        resp = client.post('/api/ai/shortest-path', json={
            'currentLat': 6.9271, 'currentLng': 79.8612,
            'faultLat':   7.2906, 'faultLng':   80.6337,
        })
        if resp.status_code != 200:
            failures.append(
                f"POST /api/ai/shortest-path returned {resp.status_code}, expected 200"
            )
        elif json_body(resp).get('success') is not True:
            failures.append("POST /api/ai/shortest-path returned a non-success envelope")

        assert not failures, (
            "Endpoints did not degrade cleanly to synthetic data with the DB down:\n  "
            + "\n  ".join(failures)
        )


# ═════════════════════════════════════════════════════════════════════════════
# 404 / 405 ERROR HANDLER TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestErrorHandlers:

    def test_unknown_route_returns_404(self, client):
        resp = client.get('/api/ai/nonexistent-endpoint')
        assert resp.status_code == 404

    def test_wrong_method_returns_405(self, client):
        """Health is GET-only; POST should return 405."""
        resp = client.post('/api/ai/health')
        assert resp.status_code == 405
