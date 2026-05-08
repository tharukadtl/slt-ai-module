"""
routes/dashboard.py — /api/ai/dashboard Blueprint
==================================================
Aggregated AI dashboard endpoint — combines forecast summary,
cluster overview, route stats, and classifier metrics into a single
payload for the admin AI Dashboard page.

Endpoints:
    GET  /api/ai/dashboard             Full dashboard payload
    GET  /api/ai/dashboard/summary     Ultra-lightweight KPI cards only
    GET  /api/ai/dashboard/classify    Predict category + priority for a description
    POST /api/ai/dashboard/retrain     Retrain all models at once

The dashboard route is designed to be called once on page load and supply
all widgets simultaneously, avoiding multiple parallel requests from the frontend.

Response shape (GET /dashboard):
    {
      success: bool,
      data: {
        forecastSummary: {
          nextWeekTotal, trend, trendPercent, accuracy, lastTrained, horizon
        },
        clusterSummary: {
          totalClusters, highRiskZones, topRegion, silhouetteScore
        },
        routeStats: {
          activeTechnicians, avgDistanceKm, avgEtaMinutes
        },
        classifierInfo: {
          catAccuracy, priAccuracy, trainedAt, method
        },
        dbStatus: {
          connected, faults, technicians, recentLocations
        },
        modelsReady: {
          forecaster, clusterer, classifier, router
        },
        generatedAt: ISO8601
      }
    }
"""

import logging
from datetime import datetime, timezone

from flask import Blueprint, jsonify, request

from config import Config, is_db_available

logger = logging.getLogger('slt_ai.routes.dashboard')

dashboard_bp = Blueprint('dashboard', __name__)

# ── Singletons ─────────────────────────────────────────────────────────────────
_forecaster = None
_clusterer  = None
_classifier = None
_router     = None


def _get_forecaster():
    global _forecaster
    if _forecaster is None:
        try:
            from models.forecasting import ProphetForecaster
            _forecaster = ProphetForecaster()
        except Exception as e:
            logger.error(f"ProphetForecaster unavailable: {e}")
    return _forecaster


def _get_clusterer():
    global _clusterer
    if _clusterer is None:
        try:
            from models.clustering import KMeansClusterer
            _clusterer = KMeansClusterer()
        except Exception as e:
            logger.error(f"KMeansClusterer unavailable: {e}")
    return _clusterer


def _get_classifier():
    global _classifier
    if _classifier is None:
        try:
            from models.classifier import FaultClassifier
            _classifier = FaultClassifier()
        except Exception as e:
            logger.error(f"FaultClassifier unavailable: {e}")
    return _classifier


def _get_router():
    global _router
    if _router is None:
        try:
            from models.route_optimizer import DijkstraRouter
            _router = DijkstraRouter()
        except Exception as e:
            logger.error(f"DijkstraRouter unavailable: {e}")
    return _router


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _ok(data, message='OK', code=200):
    return jsonify({
        'success':   True,
        'data':      data,
        'message':   message,
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }), code


def _err(message, code=500):
    logger.warning(f"Dashboard error [{code}]: {message}")
    return jsonify({
        'success':   False,
        'error':     message,
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }), code


def _safe_forecast_summary() -> dict:
    """
    Return a lightweight forecast summary — 7-day horizon only.
    Errors silently and returns a skeleton on failure.
    """
    blank = {
        'nextWeekTotal': None, 'trend': None, 'trendPercent': None,
        'accuracy': None, 'lastTrained': None, 'horizon': 7,
        'available': False,
    }
    forecaster = _get_forecaster()
    if forecaster is None:
        return blank

    try:
        from data.data_extractor import DataExtractor
        from data.synthetic_data import SyntheticDataGenerator
        from data.data_cleaner   import DataCleaner

        if is_db_available():
            raw_df = DataExtractor().get_fault_time_series(days_back=270)
        else:
            raw_df = SyntheticDataGenerator(seed=42).fault_time_series(days=270)

        if raw_df is None or raw_df.empty:
            return blank

        result = forecaster.forecast(raw_df, horizon=7)
        metrics = result.get('metrics', {})

        return {
            'nextWeekTotal': result.get('nextPeriodTotal'),
            'trend':         result.get('trend'),
            'trendPercent':  result.get('trendPercent'),
            'accuracy':      metrics.get('accuracy'),
            'lastTrained':   metrics.get('lastTrained'),
            'horizon':       7,
            'available':     True,
        }
    except Exception as exc:
        logger.warning(f"Forecast summary failed: {exc}")
        return blank


def _safe_cluster_summary() -> dict:
    """Return a cluster overview — silently handles errors."""
    blank = {
        'totalClusters': None, 'highRiskZones': None,
        'topRegion': None, 'silhouetteScore': None,
        'available': False,
    }
    clusterer = _get_clusterer()
    if clusterer is None:
        return blank

    try:
        from data.data_extractor import DataExtractor
        from data.synthetic_data import SyntheticDataGenerator

        if is_db_available():
            gps_df = DataExtractor().get_faults_with_location(days_back=90)
        else:
            gps_df = SyntheticDataGenerator(seed=42).fault_gps_points(n=500)

        if gps_df is None or gps_df.empty:
            return blank

        result   = clusterer.cluster(gps_df, n_clusters=Config.KMEANS_N_CLUSTERS)
        clusters = result.get('clusters', [])

        high_risk = sum(1 for c in clusters if c.get('riskLevel') == 'HIGH')
        top_region = clusters[0].get('regionName', 'Unknown') if clusters else 'Unknown'

        return {
            'totalClusters':  len(clusters),
            'highRiskZones':  high_risk,
            'topRegion':      top_region,
            'silhouetteScore': result.get('silhouetteScore'),
            'available':      True,
        }
    except Exception as exc:
        logger.warning(f"Cluster summary failed: {exc}")
        return blank


def _safe_route_stats() -> dict:
    """Return route optimisation stats — uses synthetic if DB unavailable."""
    blank = {
        'activeTechnicians': None, 'avgDistanceKm': None,
        'avgEtaMinutes': None, 'available': False,
    }
    router = _get_router()
    if router is None:
        return blank

    try:
        from data.data_extractor import DataExtractor
        from data.synthetic_data import SyntheticDataGenerator
        from data.feature_engineer import haversine_km, estimate_travel_minutes

        if is_db_available():
            tech_df = DataExtractor().get_available_technicians()
        else:
            tech_df = SyntheticDataGenerator(seed=42).technician_locations(n=25)

        if tech_df is None or tech_df.empty:
            return blank

        active = len(tech_df[tech_df['status'].isin(['AVAILABLE','IN_PROGRESS','TRAVELLING'])])

        # Estimate average distance from Colombo to all techs
        dists = []
        for _, row in tech_df.iterrows():
            try:
                d = haversine_km(
                    Config.SL_CENTER_LAT, Config.SL_CENTER_LNG,
                    float(row['latitude']), float(row['longitude'])
                )
                dists.append(d)
            except Exception:
                pass

        avg_dist = round(sum(dists) / len(dists), 1) if dists else None
        avg_eta  = estimate_travel_minutes(avg_dist) if avg_dist else None

        return {
            'activeTechnicians': int(active),
            'avgDistanceKm':     avg_dist,
            'avgEtaMinutes':     avg_eta,
            'available':         True,
        }
    except Exception as exc:
        logger.warning(f"Route stats failed: {exc}")
        return blank


def _safe_classifier_info() -> dict:
    """Return classifier training metadata."""
    blank = {'catAccuracy': None, 'priAccuracy': None, 'trainedAt': None, 'available': False}
    classifier = _get_classifier()
    if classifier is None:
        return blank

    try:
        meta    = classifier._meta
        metrics = meta.get('metrics', {})
        return {
            'catAccuracy': metrics.get('cat_accuracy'),
            'priAccuracy': metrics.get('pri_accuracy'),
            'trainedAt':   meta.get('trained_at'),
            'nTraining':   meta.get('training_rows'),
            'method':      'ml' if classifier._trained else 'rules',
            'available':   True,
        }
    except Exception as exc:
        logger.warning(f"Classifier info failed: {exc}")
        return blank


def _db_counts() -> dict:
    """Return record counts from MySQL, or offline indicator."""
    if not is_db_available():
        return {'connected': False}
    try:
        from data.data_extractor import DataExtractor
        counts = DataExtractor().count_records()
        return {'connected': True, **counts}
    except Exception:
        return {'connected': False}


# ─── Routes ───────────────────────────────────────────────────────────────────

@dashboard_bp.route('/dashboard', methods=['GET'])
def get_dashboard():
    """
    Full AI dashboard payload — all widgets in one request.

    GET /api/ai/dashboard
    """
    logger.info("Dashboard: building full payload")

    forecast_summary = _safe_forecast_summary()
    cluster_summary  = _safe_cluster_summary()
    route_stats      = _safe_route_stats()
    classifier_info  = _safe_classifier_info()
    db               = _db_counts()

    models_ready = {
        'forecaster': _get_forecaster() is not None,
        'clusterer':  _get_clusterer()  is not None,
        'classifier': _get_classifier() is not None,
        'router':     _get_router()     is not None,
    }

    payload = {
        'forecastSummary':  forecast_summary,
        'clusterSummary':   cluster_summary,
        'routeStats':       route_stats,
        'classifierInfo':   classifier_info,
        'dbStatus':         db,
        'modelsReady':      models_ready,
        'dataSource':       'database' if db.get('connected') else 'synthetic',
        'generatedAt':      datetime.now(timezone.utc).isoformat(),
    }

    logger.info(
        f"Dashboard served — "
        f"db={'ok' if db.get('connected') else 'offline'}, "
        f"models={sum(models_ready.values())}/4 ready"
    )
    return _ok(payload, "AI dashboard data loaded")


@dashboard_bp.route('/dashboard/summary', methods=['GET'])
def get_summary():
    """
    Ultra-lightweight KPI card data only (no heavy model calls).
    Used for rapid page-load skeleton data.

    GET /api/ai/dashboard/summary
    """
    db = _db_counts()

    return _ok({
        'dbConnected':   db.get('connected', False),
        'dbFaults':      db.get('faults'),
        'dbTechnicians': db.get('technicians'),
        'modelsReady': {
            'forecaster': _get_forecaster() is not None,
            'clusterer':  _get_clusterer()  is not None,
            'classifier': _get_classifier() is not None,
            'router':     _get_router()     is not None,
        },
        'generatedAt': datetime.now(timezone.utc).isoformat(),
    }, "Dashboard summary")


@dashboard_bp.route('/dashboard/classify', methods=['GET', 'POST'])
def classify_fault():
    """
    Predict category + priority for a fault description.
    Used by the admin portal when manually entering faults.

    GET  /api/ai/dashboard/classify?description=No+internet+since+morning
    POST /api/ai/dashboard/classify
         Body: { "description": "...", "context": { "hour": 14 } }
    """
    classifier = _get_classifier()
    if classifier is None:
        return _err('Fault classifier not available', 503)

    if request.method == 'POST':
        body        = request.get_json(silent=True) or {}
        description = body.get('description', '')
        context     = body.get('context', {})
    else:
        description = request.args.get('description', '')
        context     = {}

    description = str(description).strip()
    if not description:
        return _err("'description' is required", 400)
    if len(description) > 1000:
        return _err("Description too long (max 1000 characters)", 400)

    try:
        result = classifier.predict(description, context)
        result['inputLength'] = len(description)
        logger.debug(
            f"Classify: '{description[:50]}…' → "
            f"cat={result.get('category')}, pri={result.get('priority')}"
        )
        return _ok(result, "Fault classified")

    except Exception as exc:
        logger.exception("Classification error")
        return _err(f"Classification failed: {str(exc)}")


@dashboard_bp.route('/dashboard/retrain', methods=['POST'])
def retrain_all():
    """
    Retrain all models simultaneously.

    POST /api/ai/dashboard/retrain
    Body: { "models": ["forecaster", "clusterer", "classifier"] }  (optional)

    If 'models' is omitted, all three are retrained.
    """
    body   = request.get_json(silent=True) or {}
    target = set(body.get('models', ['forecaster', 'clusterer', 'classifier']))
    results = {}

    # ── Forecaster ─────────────────────────────────────────────────────────
    if 'forecaster' in target:
        forecaster = _get_forecaster()
        if forecaster:
            try:
                from data.data_extractor import DataExtractor
                from data.synthetic_data import SyntheticDataGenerator
                raw_df = (DataExtractor().get_fault_time_series(days_back=540)
                          if is_db_available()
                          else SyntheticDataGenerator(seed=42).fault_time_series(days=540))
                metrics = forecaster.retrain(raw_df)
                results['forecaster'] = {'status': 'retrained', 'metrics': metrics}
            except Exception as exc:
                results['forecaster'] = {'status': 'error', 'message': str(exc)}
        else:
            results['forecaster'] = {'status': 'not_available'}

    # ── Clusterer ──────────────────────────────────────────────────────────
    if 'clusterer' in target:
        clusterer = _get_clusterer()
        if clusterer:
            try:
                from data.data_extractor import DataExtractor
                from data.synthetic_data import SyntheticDataGenerator
                gps_df = (DataExtractor().get_faults_with_location(days_back=180)
                          if is_db_available()
                          else SyntheticDataGenerator(seed=42).fault_gps_points(n=1000))
                info = clusterer.retrain(gps_df)
                results['clusterer'] = {'status': 'retrained', 'info': info}
            except Exception as exc:
                results['clusterer'] = {'status': 'error', 'message': str(exc)}
        else:
            results['clusterer'] = {'status': 'not_available'}

    # ── Classifier ─────────────────────────────────────────────────────────
    if 'classifier' in target:
        classifier = _get_classifier()
        if classifier:
            try:
                metrics = classifier.train_from_db()
                results['classifier'] = {'status': 'retrained', 'metrics': metrics}
            except Exception as exc:
                results['classifier'] = {'status': 'error', 'message': str(exc)}
        else:
            results['classifier'] = {'status': 'not_available'}

    all_ok = all(r.get('status') == 'retrained' for r in results.values())
    logger.info(
        f"Retrain all: {sum(1 for r in results.values() if r.get('status')=='retrained')}"
        f"/{len(target)} succeeded"
    )
    return _ok(
        {'results': results, 'allSucceeded': all_ok},
        "Retraining complete"
    )
