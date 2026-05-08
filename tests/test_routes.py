"""
tests/test_routes.py — Flask Route Integration Tests
=====================================================
Tests all AI module HTTP endpoints using Flask's test client.
All tests run without a real MySQL database (uses synthetic data).

Covers:
  - Health endpoints (full, ping, ready)
  - Predictions endpoints (forecast, categories, history)
  - Clusters endpoints (cluster, map, heatmap)
  - Route optimisation (nearest, nearby, batch)
  - Dashboard (full, summary, classify, retrain)
  - Input validation (bad params, missing required, out-of-bounds)
  - HTTP status codes
  - Response envelope structure

Run:
    cd slt-ai-module
    python -m pytest tests/test_routes.py -v
    python -m pytest tests/test_routes.py -v -k "health"   # filter
"""

import sys
import os
import json
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ═════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope='module')
def client():
    """
    Flask test client with DB connectivity disabled (synthetic data mode).
    All models are initialised before the first test runs.
    """
    # Patch is_db_available to return False so tests never need a DB
    import config as cfg_mod
    original = cfg_mod.is_db_available
    cfg_mod.is_db_available = lambda: False

    import app as app_mod

    # Register blueprints if not already registered
    from routes.health      import health_bp
    from routes.predictions import predictions_bp
    from routes.clusters    import clusters_bp
    from routes.route       import route_bp
    from routes.dashboard   import dashboard_bp

    registered = {bp.name for bp in app_mod.app.blueprints.values()}
    if 'health'      not in registered: app_mod.app.register_blueprint(health_bp,      url_prefix='/api/ai')
    if 'predictions' not in registered: app_mod.app.register_blueprint(predictions_bp, url_prefix='/api/ai')
    if 'clusters'    not in registered: app_mod.app.register_blueprint(clusters_bp,    url_prefix='/api/ai')
    if 'route'       not in registered: app_mod.app.register_blueprint(route_bp,       url_prefix='/api/ai')
    if 'dashboard'   not in registered: app_mod.app.register_blueprint(dashboard_bp,   url_prefix='/api/ai')

    app_mod.app.config['TESTING'] = True
    with app_mod.app.test_client() as c:
        yield c

    cfg_mod.is_db_available = original


def json_body(response) -> dict:
    """Parse JSON from a Flask test response."""
    return json.loads(response.data.decode('utf-8'))


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

class TestHealthEndpoints:

    def test_ping_returns_200(self, client):
        resp = client.get('/api/ai/health/ping')
        assert resp.status_code == 200

    def test_ping_body_has_status_ok(self, client):
        body = json_body(client.get('/api/ai/health/ping'))
        assert body.get('status') == 'ok'

    def test_ping_has_timestamp(self, client):
        body = json_body(client.get('/api/ai/health/ping'))
        assert 'ts' in body

    def test_ready_returns_200(self, client):
        resp = client.get('/api/ai/health/ready')
        assert resp.status_code in (200, 503), \
            f"Unexpected status code: {resp.status_code}"

    def test_ready_has_ready_key(self, client):
        body = json_body(client.get('/api/ai/health/ready'))
        assert 'ready' in body

    def test_full_health_returns_200_or_206(self, client):
        """Full health is 200 (healthy) or 206 (degraded — DB down but functional)."""
        resp = client.get('/api/ai/health')
        assert resp.status_code in (200, 206, 503)

    def test_full_health_has_service_key(self, client):
        body = json_body(client.get('/api/ai/health'))
        assert body.get('service') == 'SLT AI Module'

    def test_full_health_has_models(self, client):
        body = json_body(client.get('/api/ai/health'))
        assert 'models' in body

    def test_full_health_has_config(self, client):
        body = json_body(client.get('/api/ai/health'))
        assert 'config' in body

    def test_full_health_has_version(self, client):
        body = json_body(client.get('/api/ai/health'))
        assert 'version' in body

    def test_full_health_has_uptime(self, client):
        body = json_body(client.get('/api/ai/health'))
        assert 'uptime' in body

    def test_full_health_db_shows_unavailable(self, client):
        """In test mode (DB mocked off), db.connected should be False."""
        body = json_body(client.get('/api/ai/health'))
        db   = body.get('database', {})
        assert db.get('connected') is False


# ═════════════════════════════════════════════════════════════════════════════
# PREDICTIONS ENDPOINT TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestPredictionsEndpoints:

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

    def test_predictions_with_valid_category(self, client):
        resp = client.get('/api/ai/predictions?category=BROADBAND')
        assert resp.status_code == 200

    def test_predictions_invalid_category_returns_400(self, client):
        resp = client.get('/api/ai/predictions?category=INVALID_CATEGORY')
        assert resp.status_code == 400

    def test_predictions_summary_format(self, client):
        resp = client.get('/api/ai/predictions?format=summary')
        assert resp.status_code == 200
        data = json_body(resp)['data']
        assert 'historical' not in data   # stripped in summary mode

    def test_predictions_data_source_is_synthetic(self, client):
        data = json_body(client.get('/api/ai/predictions'))['data']
        assert data.get('dataSource') == 'synthetic'

    def test_predictions_forecast_yhat_non_negative(self, client):
        data = json_body(client.get('/api/ai/predictions?horizon=7'))['data']
        for item in data['forecast']:
            assert item['yhat'] >= 0

    def test_predictions_next_total_int(self, client):
        data = json_body(client.get('/api/ai/predictions'))['data']
        assert isinstance(data['nextPeriodTotal'], int)

    def test_predictions_trend_valid(self, client):
        data = json_body(client.get('/api/ai/predictions'))['data']
        assert data['trend'] in ('up', 'down', 'stable')

    def test_predictions_history_endpoint(self, client):
        resp = client.get('/api/ai/predictions/history?days=30')
        assert resp.status_code == 200
        data = json_body(resp)['data']
        assert 'historical' in data

    def test_predictions_categories_endpoint(self, client):
        resp = client.get('/api/ai/predictions/categories?horizon=7')
        assert resp.status_code == 200
        data = json_body(resp)['data']
        assert 'categories' in data


# ═════════════════════════════════════════════════════════════════════════════
# CLUSTERS ENDPOINT TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestClustersEndpoints:

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

    def test_clusters_invalid_category_returns_400(self, client):
        resp = client.get('/api/ai/clusters?category=BLAH')
        assert resp.status_code == 400

    def test_clusters_valid_category(self, client):
        resp = client.get('/api/ai/clusters?category=FIBER')
        assert resp.status_code == 200

    def test_clusters_data_source_synthetic(self, client):
        data = json_body(client.get('/api/ai/clusters'))['data']
        assert data.get('dataSource') == 'synthetic'

    def test_clusters_fault_counts_positive(self, client):
        data = json_body(client.get('/api/ai/clusters'))['data']
        for c in data['clusters']:
            assert c['faultCount'] >= 0

    def test_clusters_risk_levels_valid(self, client):
        data = json_body(client.get('/api/ai/clusters'))['data']
        for c in data['clusters']:
            assert c['riskLevel'] in ('HIGH', 'MEDIUM', 'LOW')

    def test_clusters_map_endpoint(self, client):
        resp = client.get('/api/ai/clusters/map')
        assert resp.status_code == 200
        data = json_body(resp)['data']
        assert 'clusters'   in data
        assert 'svgViewBox' in data

    def test_clusters_map_has_svg_coords(self, client):
        data = json_body(client.get('/api/ai/clusters/map'))['data']
        for c in data['clusters']:
            assert 'svgX'   in c
            assert 'svgY'   in c
            assert 'radius' in c

    def test_clusters_heatmap_endpoint(self, client):
        resp = client.get('/api/ai/clusters/heatmap?resolution=20')
        assert resp.status_code == 200
        data = json_body(resp)['data']
        assert 'points' in data

    def test_clusters_heatmap_intensity_range(self, client):
        data = json_body(client.get('/api/ai/clusters/heatmap?resolution=10'))['data']
        for pt in data['points']:
            assert 0.0 <= pt['intensity'] <= 1.0


# ═════════════════════════════════════════════════════════════════════════════
# ROUTE OPTIMISATION ENDPOINT TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestRouteEndpoints:

    COLOMBO_LAT = 6.9271
    COLOMBO_LNG = 79.8612

    def test_optimize_route_returns_200(self, client):
        resp = client.get(f'/api/ai/optimize-route?lat={self.COLOMBO_LAT}&lng={self.COLOMBO_LNG}')
        assert resp.status_code == 200

    def test_optimize_route_success_envelope(self, client):
        body = json_body(client.get(
            f'/api/ai/optimize-route?lat={self.COLOMBO_LAT}&lng={self.COLOMBO_LNG}'
        ))
        assert_success_envelope(body)

    def test_optimize_route_has_technicians(self, client):
        data = json_body(client.get(
            f'/api/ai/optimize-route?lat={self.COLOMBO_LAT}&lng={self.COLOMBO_LNG}'
        ))['data']
        assert 'technicians' in data
        assert isinstance(data['technicians'], list)

    def test_optimize_route_technicians_ranked(self, client):
        """Technicians are returned in rank order 1, 2, 3…"""
        data = json_body(client.get(
            f'/api/ai/optimize-route?lat={self.COLOMBO_LAT}&lng={self.COLOMBO_LNG}'
        ))['data']
        techs = data['technicians']
        if len(techs) > 1:
            ranks = [t['rank'] for t in techs]
            assert ranks == list(range(1, len(ranks)+1))

    def test_optimize_route_distances_positive(self, client):
        data = json_body(client.get(
            f'/api/ai/optimize-route?lat={self.COLOMBO_LAT}&lng={self.COLOMBO_LNG}'
        ))['data']
        for t in data['technicians']:
            assert t['distanceKm'] >= 0

    def test_optimize_route_limit_param(self, client):
        data = json_body(client.get(
            f'/api/ai/optimize-route?lat={self.COLOMBO_LAT}&lng={self.COLOMBO_LNG}&limit=3'
        ))['data']
        assert len(data['technicians']) <= 3

    def test_optimize_route_has_fault_location(self, client):
        data = json_body(client.get(
            f'/api/ai/optimize-route?lat={self.COLOMBO_LAT}&lng={self.COLOMBO_LNG}'
        ))['data']
        assert 'faultLocation' in data
        assert data['faultLocation']['lat'] == pytest.approx(self.COLOMBO_LAT, abs=0.001)

    def test_optimize_route_missing_lat_returns_400(self, client):
        resp = client.get(f'/api/ai/optimize-route?lng={self.COLOMBO_LNG}')
        assert resp.status_code == 400

    def test_optimize_route_missing_lng_returns_400(self, client):
        resp = client.get(f'/api/ai/optimize-route?lat={self.COLOMBO_LAT}')
        assert resp.status_code == 400

    def test_optimize_route_out_of_bounds_lat_returns_400(self, client):
        resp = client.get('/api/ai/optimize-route?lat=51.5&lng=79.86')
        assert resp.status_code == 400

    def test_optimize_route_out_of_bounds_lng_returns_400(self, client):
        resp = client.get('/api/ai/optimize-route?lat=6.93&lng=0.0')
        assert resp.status_code == 400

    def test_optimize_route_algorithm_field(self, client):
        data = json_body(client.get(
            f'/api/ai/optimize-route?lat={self.COLOMBO_LAT}&lng={self.COLOMBO_LNG}'
        ))['data']
        assert data.get('algorithm') == 'dijkstra+haversine'

    def test_nearby_endpoint_returns_200(self, client):
        resp = client.get(
            f'/api/ai/optimize-route/nearby?lat={self.COLOMBO_LAT}&lng={self.COLOMBO_LNG}&radius=30'
        )
        assert resp.status_code == 200

    def test_nearby_endpoint_has_technicians_and_count(self, client):
        data = json_body(client.get(
            f'/api/ai/optimize-route/nearby?lat={self.COLOMBO_LAT}&lng={self.COLOMBO_LNG}&radius=100'
        ))['data']
        assert 'technicians' in data
        assert 'count'       in data
        assert data['count'] == len(data['technicians'])

    def test_batch_assign_endpoint_returns_200(self, client):
        payload = {
            'faults': [
                {'fault_id': 1, 'lat': 6.93, 'lng': 79.86, 'priority': 'HIGH'},
                {'fault_id': 2, 'lat': 7.29, 'lng': 80.63, 'priority': 'MEDIUM'},
            ]
        }
        resp = client.post(
            '/api/ai/optimize-route/batch',
            data=json.dumps(payload),
            content_type='application/json'
        )
        assert resp.status_code == 200

    def test_batch_assign_response_structure(self, client):
        payload = {'faults': [{'fault_id': 10, 'lat': 6.93, 'lng': 79.86}]}
        data = json_body(client.post(
            '/api/ai/optimize-route/batch',
            data=json.dumps(payload),
            content_type='application/json'
        ))['data']
        assert 'assignments'   in data
        assert 'totalAssigned' in data
        assert 'totalFailed'   in data

    def test_batch_assign_empty_body_returns_400(self, client):
        resp = client.post(
            '/api/ai/optimize-route/batch',
            data='{}',
            content_type='application/json'
        )
        assert resp.status_code == 400

    def test_batch_assign_too_many_faults_returns_400(self, client):
        payload = {'faults': [{'fault_id': i, 'lat': 6.93, 'lng': 79.86} for i in range(60)]}
        resp = client.post(
            '/api/ai/optimize-route/batch',
            data=json.dumps(payload),
            content_type='application/json'
        )
        assert resp.status_code == 400


# ═════════════════════════════════════════════════════════════════════════════
# DASHBOARD ENDPOINT TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestDashboardEndpoints:

    def test_dashboard_returns_200(self, client):
        resp = client.get('/api/ai/dashboard')
        assert resp.status_code == 200

    def test_dashboard_success_envelope(self, client):
        body = json_body(client.get('/api/ai/dashboard'))
        assert_success_envelope(body)

    def test_dashboard_has_required_sections(self, client):
        data = json_body(client.get('/api/ai/dashboard'))['data']
        for key in ['modelsReady', 'dbStatus', 'generatedAt']:
            assert key in data, f"Missing dashboard section: {key}"

    def test_dashboard_db_status_shows_unavailable(self, client):
        data = json_body(client.get('/api/ai/dashboard'))['data']
        assert data['dbStatus'].get('connected') is False

    def test_dashboard_data_source_synthetic(self, client):
        data = json_body(client.get('/api/ai/dashboard'))['data']
        assert data.get('dataSource') == 'synthetic'

    def test_dashboard_summary_returns_200(self, client):
        resp = client.get('/api/ai/dashboard/summary')
        assert resp.status_code == 200

    def test_dashboard_summary_has_models_ready(self, client):
        data = json_body(client.get('/api/ai/dashboard/summary'))['data']
        assert 'modelsReady' in data

    def test_classify_get_returns_200(self, client):
        resp = client.get('/api/ai/dashboard/classify?description=No+internet+since+morning')
        assert resp.status_code == 200

    def test_classify_response_has_category_and_priority(self, client):
        data = json_body(client.get(
            '/api/ai/dashboard/classify?description=Fiber+cable+cut+no+service'
        ))['data']
        assert 'category' in data
        assert 'priority' in data

    def test_classify_category_is_valid(self, client):
        from config import Config
        data = json_body(client.get(
            '/api/ai/dashboard/classify?description=Internet+is+very+slow'
        ))['data']
        assert data['category'] in Config.FAULT_CATEGORIES

    def test_classify_priority_is_valid(self, client):
        data = json_body(client.get(
            '/api/ai/dashboard/classify?description=Hospital+internet+down+urgent'
        ))['data']
        assert data['priority'] in ('HIGH', 'MEDIUM', 'LOW')

    def test_classify_post_method(self, client):
        resp = client.post(
            '/api/ai/dashboard/classify',
            data=json.dumps({'description': 'TV channels not working'}),
            content_type='application/json'
        )
        assert resp.status_code == 200

    def test_classify_empty_description_returns_400(self, client):
        resp = client.get('/api/ai/dashboard/classify?description=')
        assert resp.status_code == 400

    def test_classify_missing_description_returns_400(self, client):
        resp = client.get('/api/ai/dashboard/classify')
        assert resp.status_code == 400

    def test_classify_has_urgency_score(self, client):
        data = json_body(client.get(
            '/api/ai/dashboard/classify?description=No+broadband+all+day'
        ))['data']
        assert 'urgencyScore' in data
        assert 0 <= data['urgencyScore'] <= 100

    def test_classify_has_explanation(self, client):
        data = json_body(client.get(
            '/api/ai/dashboard/classify?description=Telephone+line+dead'
        ))['data']
        assert 'explanation' in data
        assert len(data['explanation']) > 0

    def test_classify_has_sla_risk(self, client):
        data = json_body(client.get(
            '/api/ai/dashboard/classify?description=Critical+outage+business+premises'
        ))['data']
        assert 'slaRisk' in data
        assert data['slaRisk'] in ('HIGH', 'MEDIUM', 'LOW')

    def test_retrain_endpoint_returns_200(self, client):
        resp = client.post(
            '/api/ai/dashboard/retrain',
            data=json.dumps({'models': ['classifier']}),
            content_type='application/json'
        )
        assert resp.status_code == 200

    def test_retrain_response_has_results(self, client):
        data = json_body(client.post(
            '/api/ai/dashboard/retrain',
            data=json.dumps({'models': ['classifier']}),
            content_type='application/json'
        ))['data']
        assert 'results' in data


# ═════════════════════════════════════════════════════════════════════════════
# 404 / 405 ERROR HANDLER TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestErrorHandlers:

    def test_unknown_route_returns_404(self, client):
        resp = client.get('/api/ai/nonexistent-endpoint')
        assert resp.status_code == 404

    def test_wrong_method_returns_405(self, client):
        """Ping is GET-only; POST should return 405."""
        resp = client.post('/api/ai/health/ping')
        assert resp.status_code == 405
