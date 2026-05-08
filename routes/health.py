"""
routes/health.py — /api/ai/health Blueprint
============================================
Service health check endpoint.

Endpoints:
    GET  /api/ai/health          Full health status
    GET  /api/ai/health/ping     Ultra-lightweight liveness probe
    GET  /api/ai/health/ready    Readiness check (DB + models loaded)

Returns:
    - Service version and uptime
    - Database connectivity and record counts
    - Model loading status for all three models
    - System resource snapshot
    - Configuration summary

HTTP status codes:
    200  All systems operational
    206  Degraded (DB unavailable, using synthetic data — still functional)
    503  Critical failure (Flask itself broken — should not happen)
"""

import os
import time
import platform
import logging
from datetime import datetime, timezone

from flask import Blueprint, jsonify

from config import Config, is_db_available

logger = logging.getLogger('slt_ai.routes.health')

# ── Module-level start time (for uptime calculation) ──────────────────────────
_SERVICE_START = time.time()
_SERVICE_VERSION = '1.0.0'

health_bp = Blueprint('health', __name__)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _uptime_str() -> str:
    """Human-readable uptime since Flask app started."""
    elapsed = int(time.time() - _SERVICE_START)
    h, rem  = divmod(elapsed, 3600)
    m, s    = divmod(rem, 60)
    return f"{h}h {m}m {s}s"


def _model_status() -> dict:
    """
    Check whether each model class is importable and its saved file exists.
    Avoids actually loading the models (expensive); just checks file presence.
    """
    import pathlib
    model_dir = pathlib.Path(Config.MODEL_DIR)

    statuses = {}

    # Prophet forecaster
    try:
        from models.forecasting import ProphetForecaster, MODEL_SAVE_PATH, _PROPHET_AVAILABLE
        statuses['prophet_forecaster'] = {
            'importable':     True,
            'prophet_lib':    _PROPHET_AVAILABLE,
            'model_saved':    MODEL_SAVE_PATH.exists(),
            'status':         'ready' if _PROPHET_AVAILABLE else 'degraded (prophet not installed)',
        }
    except Exception as e:
        statuses['prophet_forecaster'] = {'importable': False, 'error': str(e), 'status': 'error'}

    # K-Means clusterer
    try:
        from models.clustering import KMeansClusterer, MODEL_SAVE_PATH as CL_PATH, _SKL_AVAILABLE
        statuses['kmeans_clusterer'] = {
            'importable':  True,
            'sklearn_lib': _SKL_AVAILABLE,
            'model_saved': CL_PATH.exists(),
            'status':      'ready' if _SKL_AVAILABLE else 'degraded (scikit-learn not installed)',
        }
    except Exception as e:
        statuses['kmeans_clusterer'] = {'importable': False, 'error': str(e), 'status': 'error'}

    # Fault classifier
    try:
        from models.classifier import FaultClassifier, CAT_MODEL_PATH, _SKL_AVAILABLE as CL_SKL
        statuses['fault_classifier'] = {
            'importable':  True,
            'sklearn_lib': CL_SKL,
            'model_saved': CAT_MODEL_PATH.exists(),
            'status':      'ready' if CL_SKL else 'degraded (scikit-learn not installed)',
        }
    except Exception as e:
        statuses['fault_classifier'] = {'importable': False, 'error': str(e), 'status': 'error'}

    # Dijkstra router
    try:
        from models.route_optimizer import DijkstraRouter
        statuses['dijkstra_router'] = {
            'importable': True,
            'status':     'ready',   # pure Python, no external deps beyond pandas
        }
    except Exception as e:
        statuses['dijkstra_router'] = {'importable': False, 'error': str(e), 'status': 'error'}

    return statuses


def _db_status() -> dict:
    """Return DB connectivity info and record counts."""
    db_ok = is_db_available()
    if not db_ok:
        return {
            'connected': False,
            'host':      f"{Config.DB_HOST}:{Config.DB_PORT}",
            'database':  Config.DB_NAME,
            'message':   'Database unavailable — AI module running on synthetic data',
        }

    try:
        from data.data_extractor import DataExtractor
        extractor = DataExtractor()
        counts    = extractor.count_records()
    except Exception as e:
        counts = {'error': str(e)}

    return {
        'connected': True,
        'host':      f"{Config.DB_HOST}:{Config.DB_PORT}",
        'database':  Config.DB_NAME,
        'counts':    counts,
    }


def _system_info() -> dict:
    """Lightweight system resource snapshot."""
    info = {
        'platform':    platform.system(),
        'python':      platform.python_version(),
        'pid':         os.getpid(),
    }
    # psutil is optional — skip if not installed
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        info['memory_mb']   = round(proc.memory_info().rss / 1024 / 1024, 1)
        info['cpu_percent'] = psutil.cpu_percent(interval=0.1)
    except ImportError:
        pass
    return info


# ─── Routes ───────────────────────────────────────────────────────────────────

@health_bp.route('/health', methods=['GET'])
def full_health():
    """
    Full health status report.

    Returns 200 if DB is available and all models are importable.
    Returns 206 if DB is unavailable (synthetic data fallback — still useful).
    """
    db      = _db_status()
    models  = _model_status()
    sys_inf = _system_info()

    # Overall status
    all_models_ok = all(
        m.get('status', '') in ('ready', 'degraded (prophet not installed)',
                                'degraded (scikit-learn not installed)')
        for m in models.values()
    )

    if db['connected'] and all_models_ok:
        overall_status = 'healthy'
        http_code      = 200
    elif all_models_ok:
        overall_status = 'degraded'          # DB down but models work on synthetic data
        http_code      = 206
    else:
        overall_status = 'unhealthy'
        http_code      = 503

    payload = {
        'service':     'SLT AI Module',
        'version':     _SERVICE_VERSION,
        'status':      overall_status,
        'uptime':      _uptime_str(),
        'timestamp':   datetime.now(timezone.utc).isoformat(),
        'database':    db,
        'models':      models,
        'config': {
            'forecast_horizon_days':      Config.FORECAST_HORIZON_DAYS,
            'forecast_min_history_days':  Config.FORECAST_MIN_HISTORY_DAYS,
            'kmeans_clusters':            Config.KMEANS_N_CLUSTERS,
            'route_radius_km':            Config.ROUTE_SEARCH_RADIUS_KM,
            'retrain_interval_hours':     Config.RETRAIN_INTERVAL_HOURS,
            'sl_bounds': {
                'lat': [Config.SL_LAT_MIN, Config.SL_LAT_MAX],
                'lng': [Config.SL_LNG_MIN, Config.SL_LNG_MAX],
            },
        },
        'system': sys_inf,
    }

    logger.debug(f"Health check — status={overall_status}, http={http_code}")
    return jsonify(payload), http_code


@health_bp.route('/health/ping', methods=['GET'])
def ping():
    """
    Ultra-lightweight liveness probe — used by Docker/Kubernetes health checks.
    Returns 200 immediately with minimal payload.
    """
    return jsonify({
        'status': 'ok',
        'ts':     datetime.now(timezone.utc).isoformat(),
    }), 200


@health_bp.route('/health/ready', methods=['GET'])
def readiness():
    """
    Readiness check — confirms the service can handle requests.
    Checks DB connectivity and that at least the classifier is importable.
    Returns 200 if ready, 503 if not.
    """
    db_ok  = is_db_available()
    models = _model_status()

    any_model_ready = any(
        m.get('importable', False) for m in models.values()
    )

    if any_model_ready:
        return jsonify({
            'ready':      True,
            'db':         db_ok,
            'dataSource': 'database' if db_ok else 'synthetic',
            'ts':         datetime.now(timezone.utc).isoformat(),
        }), 200

    return jsonify({
        'ready':   False,
        'message': 'No models importable — check logs',
        'ts':      datetime.now(timezone.utc).isoformat(),
    }), 503
