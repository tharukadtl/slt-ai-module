"""
app.py — SLT AI Module Flask Application
=========================================
Main entry point for the Python Flask AI microservice.

Endpoints:
    GET  /api/ai/health             — Service health + DB status
    GET  /api/ai/predictions        — Prophet fault volume forecast
    GET  /api/ai/clusters           — K-Means geographic clusters
    GET  /api/ai/optimize-route     — Dijkstra nearest-technician routing
    GET  /api/ai/dashboard          — Combined dashboard data
    POST /api/ai/retrain            — Trigger model retraining

Run:
    python app.py                    (development)
    gunicorn -w 2 -b 0.0.0.0:5000 app:app   (production)
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Any

from flask import Flask, jsonify, request
from flask_cors import CORS

from config import Config, logger, is_db_available

# ─── Import models (lazy — they initialise on first call) ────────────────────
# Models live in models/ directory (next to be built)
# We use try/except so app starts even if a model file is missing
try:
    from models.forecasting import ProphetForecaster
    _has_forecaster = True
except ImportError as e:
    logger.warning(f"ProphetForecaster not available: {e}")
    _has_forecaster = False

try:
    from models.clustering import KMeansClusterer
    _has_clusterer = True
except ImportError as e:
    logger.warning(f"KMeansClusterer not available: {e}")
    _has_clusterer = False

try:
    from models.route_optimizer import DijkstraRouter
    _has_router = True
except ImportError as e:
    logger.warning(f"DijkstraRouter not available: {e}")
    _has_router = False

# ─── Data layer ───────────────────────────────────────────────────────────────
from data.data_extractor import DataExtractor
from data.synthetic_data import SyntheticDataGenerator

# ─── App init ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['SECRET_KEY'] = Config.SECRET_KEY
app.config['JSON_SORT_KEYS'] = False

# CORS — allow Spring Boot admin portal and React dev server
CORS(app, resources={
    r"/api/*": {
        "origins": [
            "http://localhost:3000",
            "http://localhost:3001",
            "http://localhost:8080",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:8080",
        ]
    }
})

# ─── Singletons ───────────────────────────────────────────────────────────────
_extractor  = DataExtractor()
_synth      = SyntheticDataGenerator(seed=42)
_forecaster = ProphetForecaster()   if _has_forecaster else None
_clusterer  = KMeansClusterer()     if _has_clusterer  else None
_router     = DijkstraRouter()      if _has_router     else None


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def ok(data: Any, message: str = "OK") -> tuple:
    return jsonify({
        "success":   True,
        "data":      data,
        "message":   message,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }), 200


def err(message: str, status: int = 500) -> tuple:
    return jsonify({
        "success":   False,
        "error":     message,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }), status


def _int_param(name: str, default: int, minimum: int = 1, maximum: int = 9999) -> int:
    try:
        val = int(request.args.get(name, default))
        return max(minimum, min(maximum, val))
    except (ValueError, TypeError):
        return default


def _float_param(name: str, default: float) -> float:
    try:
        return float(request.args.get(name, default))
    except (ValueError, TypeError):
        return default


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/ai/health', methods=['GET'])
def health():
    """
    Health check endpoint.
    Returns service status, DB availability, and model readiness.
    """
    db_ok     = is_db_available()
    db_counts = _extractor.count_records() if db_ok else {'db_available': False}

    payload = {
        "service":    "SLT AI Module",
        "version":    "1.0.0",
        "status":     "healthy",
        "timestamp":  datetime.utcnow().isoformat() + "Z",
        "db_status":  "connected" if db_ok else "unavailable (using synthetic data)",
        "db_counts":  db_counts,
        "models": {
            "prophet_forecaster": "available" if _has_forecaster else "not loaded",
            "kmeans_clusterer":   "available" if _has_clusterer  else "not loaded",
            "dijkstra_router":    "available" if _has_router     else "not loaded",
        },
        "config": {
            "forecast_horizon_days":    Config.FORECAST_HORIZON_DAYS,
            "kmeans_clusters":          Config.KMEANS_N_CLUSTERS,
            "route_radius_km":          Config.ROUTE_SEARCH_RADIUS_KM,
            "min_history_days":         Config.FORECAST_MIN_HISTORY_DAYS,
        }
    }
    return jsonify(payload), 200


@app.route('/api/ai/predictions', methods=['GET'])
def predictions():
    """
    Prophet time-series fault volume forecast.

    Query params:
        horizon (int): Forecast horizon in days (default 30, max 90)
        category (str): Filter by fault category (optional)

    Returns:
        {
          historical: [{ds, y, actual}],
          forecast:   [{ds, yhat, yhat_lower, yhat_upper, isForecast}],
          metrics:    {mae, rmse, accuracy, trainingDays, horizon, lastTrained},
          dataSource: "database" | "synthetic"
        }
    """
    if not _has_forecaster or _forecaster is None:
        return err("Prophet forecasting model not available", 503)

    horizon = _int_param('horizon', Config.FORECAST_HORIZON_DAYS, 7, 90)
    category = request.args.get('category', None)

    try:
        # Try live DB data first
        raw_df = None
        data_source = 'synthetic'

        if is_db_available():
            raw_df = _extractor.get_fault_time_series(
                days_back=max(Config.FORECAST_MIN_HISTORY_DAYS * 2, 540)
            )
            if raw_df is not None and len(raw_df) >= Config.FORECAST_MIN_HISTORY_DAYS:
                data_source = 'database'
                logger.info(f"Using DB data for forecast ({len(raw_df)} days)")
            else:
                raw_df = None

        # Fall back to synthetic
        if raw_df is None:
            logger.info("Using synthetic data for forecast")
            raw_df = _synth.fault_time_series(days=540)
            data_source = 'synthetic'

        result = _forecaster.forecast(raw_df, horizon=horizon)
        result['dataSource'] = data_source
        return ok(result, f"{horizon}-day fault volume forecast")

    except Exception as exc:
        logger.exception("Forecast error")
        return err(f"Forecast failed: {str(exc)}")


@app.route('/api/ai/clusters', methods=['GET'])
def clusters():
    """
    K-Means geographic fault demand clusters.

    Query params:
        n_clusters (int): Number of clusters (default 5, range 2–10)
        days (int):       History window in days (default 180)

    Returns:
        {
          clusters: [{
            clusterId, regionName, faultCount, riskLevel,
            centroid: {lat, lng},
            density, topCategory, color
          }],
          totalFaults, dataSource
        }
    """
    if not _has_clusterer or _clusterer is None:
        return err("K-Means clustering model not available", 503)

    n_clusters = _int_param('n_clusters', Config.KMEANS_N_CLUSTERS, 2, 10)
    days       = _int_param('days', 180, 30, 730)

    try:
        gps_df      = None
        data_source = 'synthetic'

        if is_db_available():
            gps_df = _extractor.get_faults_with_location(days_back=days)
            if gps_df is not None and len(gps_df) >= 20:
                data_source = 'database'
            else:
                gps_df = None

        if gps_df is None:
            logger.info("Using synthetic GPS data for clustering")
            gps_df      = _synth.fault_gps_points(n=1000, days_back=days)
            data_source = 'synthetic'

        result = _clusterer.cluster(gps_df, n_clusters=n_clusters)
        result['dataSource'] = data_source
        return ok(result, f"K-Means clustering: {n_clusters} clusters")

    except Exception as exc:
        logger.exception("Clustering error")
        return err(f"Clustering failed: {str(exc)}")


@app.route('/api/ai/optimize-route', methods=['GET'])
def optimize_route():
    """
    Dijkstra + Haversine nearest-technician route optimisation.

    Query params:
        lat   (float): Fault latitude  (required)
        lng   (float): Fault longitude (required)
        limit (int):   Max technicians to return (default 5)

    Returns:
        {
          faultLocation: {lat, lng},
          technicians: [{
            technicianId, technicianName, phone, branchName,
            distanceKm, estimatedArrivalMinutes, status,
            currentJobId, rank, latitude, longitude
          }],
          algorithm: "dijkstra+haversine",
          dataSource
        }
    """
    if not _has_router or _router is None:
        return err("Route optimisation model not available", 503)

    fault_lat = _float_param('lat', 0.0)
    fault_lng = _float_param('lng', 0.0)
    limit     = _int_param('limit', 5, 1, 20)

    # Validate fault coordinates
    if not (Config.SL_LAT_MIN <= fault_lat <= Config.SL_LAT_MAX and
            Config.SL_LNG_MIN <= fault_lng <= Config.SL_LNG_MAX):
        return err(
            f"Fault coordinates ({fault_lat}, {fault_lng}) outside Sri Lanka bounds. "
            f"Expected lat {Config.SL_LAT_MIN}–{Config.SL_LAT_MAX}, "
            f"lng {Config.SL_LNG_MIN}–{Config.SL_LNG_MAX}.",
            400
        )

    try:
        tech_df     = None
        data_source = 'synthetic'

        if is_db_available():
            tech_df = _extractor.get_available_technicians()
            if tech_df is not None and len(tech_df) > 0:
                data_source = 'database'
            else:
                tech_df = None

        if tech_df is None:
            logger.info("Using synthetic technician data for routing")
            tech_df     = _synth.technician_locations(n=25)
            data_source = 'synthetic'

        result = _router.find_nearest(
            technicians_df=tech_df,
            fault_lat=fault_lat,
            fault_lng=fault_lng,
            limit=limit
        )
        result['dataSource']     = data_source
        result['faultLocation']  = {'lat': fault_lat, 'lng': fault_lng}
        return ok(result, f"Found {len(result.get('technicians', []))} nearest technicians")

    except Exception as exc:
        logger.exception("Route optimisation error")
        return err(f"Route optimisation failed: {str(exc)}")


@app.route('/api/ai/dashboard', methods=['GET'])
def dashboard():
    """
    Combined dashboard data for the admin AI Dashboard page.
    Aggregates forecast summary, cluster overview, and quick stats.
    Returns lightweight data suitable for dashboard widgets.
    """
    try:
        # Quick forecast (7 days, lightweight)
        forecast_summary = None
        if _has_forecaster and _forecaster is not None:
            try:
                raw_df = (
                    _extractor.get_fault_time_series(days_back=270)
                    if is_db_available()
                    else _synth.fault_time_series(days=270)
                )
                if raw_df is not None:
                    forecast_full    = _forecaster.forecast(raw_df, horizon=7)
                    forecast_summary = {
                        'nextWeekTotal':  forecast_full.get('nextPeriodTotal'),
                        'trend':          forecast_full.get('trend'),
                        'accuracy':       forecast_full.get('metrics', {}).get('accuracy'),
                        'lastTrained':    forecast_full.get('metrics', {}).get('lastTrained'),
                    }
            except Exception as fe:
                logger.warning(f"Dashboard forecast mini-error: {fe}")

        # Cluster overview
        cluster_summary = None
        if _has_clusterer and _clusterer is not None:
            try:
                gps_df = (
                    _extractor.get_faults_with_location(days_back=90)
                    if is_db_available()
                    else _synth.fault_gps_points(n=500, days_back=90)
                )
                if gps_df is not None:
                    cdata = _clusterer.cluster(gps_df, n_clusters=Config.KMEANS_N_CLUSTERS)
                    cluster_summary = {
                        'totalClusters':  len(cdata.get('clusters', [])),
                        'highRiskZones':  sum(
                            1 for c in cdata.get('clusters', [])
                            if c.get('riskLevel') == 'HIGH'
                        ),
                        'topRegion': (cdata.get('clusters', [{}])[0].get('regionName') or 'Unknown')
                                     if cdata.get('clusters') else 'Unknown',
                    }
            except Exception as ce:
                logger.warning(f"Dashboard cluster mini-error: {ce}")

        db_counts = _extractor.count_records() if is_db_available() else {}

        payload = {
            'dbAvailable':    is_db_available(),
            'dbCounts':       db_counts,
            'forecastSummary': forecast_summary,
            'clusterSummary':  cluster_summary,
            'modelsReady': {
                'forecaster': _has_forecaster,
                'clusterer':  _has_clusterer,
                'router':     _has_router,
            },
            'generatedAt': datetime.utcnow().isoformat() + "Z",
        }
        return ok(payload, "Dashboard data loaded")

    except Exception as exc:
        logger.exception("Dashboard error")
        return err(f"Dashboard failed: {str(exc)}")


@app.route('/api/ai/retrain', methods=['POST'])
def retrain():
    """
    Trigger model retraining with latest DB data.
    POST body: { "models": ["forecaster", "clusterer"] }  (optional, default all)
    """
    body       = request.get_json(silent=True) or {}
    target_models = body.get('models', ['forecaster', 'clusterer'])
    results    = {}

    if 'forecaster' in target_models and _has_forecaster and _forecaster is not None:
        try:
            raw_df = (
                _extractor.get_fault_time_series(days_back=540)
                if is_db_available()
                else _synth.fault_time_series(days=540)
            )
            metrics = _forecaster.retrain(raw_df)
            results['forecaster'] = {'status': 'retrained', 'metrics': metrics}
        except Exception as exc:
            results['forecaster'] = {'status': 'error', 'message': str(exc)}

    if 'clusterer' in target_models and _has_clusterer and _clusterer is not None:
        try:
            gps_df = (
                _extractor.get_faults_with_location(days_back=180)
                if is_db_available()
                else _synth.fault_gps_points(n=1000)
            )
            info = _clusterer.retrain(gps_df)
            results['clusterer'] = {'status': 'retrained', 'info': info}
        except Exception as exc:
            results['clusterer'] = {'status': 'error', 'message': str(exc)}

    return ok(results, "Retraining complete")


# ─────────────────────────────────────────────────────────────────────────────
# ERROR HANDLERS
# ─────────────────────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(_):
    return err("Endpoint not found", 404)


@app.errorhandler(405)
def method_not_allowed(_):
    return err("Method not allowed", 405)


@app.errorhandler(500)
def internal_error(exc):
    logger.error(f"Unhandled exception: {exc}")
    return err("Internal server error", 500)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("  SLT AI Module — Starting Flask Server")
    logger.info(f"  Host:    {Config.HOST}:{Config.PORT}")
    logger.info(f"  Debug:   {Config.DEBUG}")
    logger.info(f"  DB:      {Config.DB_HOST}:{Config.DB_PORT}/{Config.DB_NAME}")
    logger.info("=" * 60)

    # Ensure model save directory exists
    import pathlib
    pathlib.Path(Config.MODEL_DIR).mkdir(parents=True, exist_ok=True)

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False,
        use_reloader=False
    )
