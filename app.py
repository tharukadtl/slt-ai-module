"""
app.py — SLT AI Module Flask Application
=========================================
Main entry point for the Python Flask AI microservice.

Endpoints:
    GET  /api/ai/health             — Service health + DB status
    GET  /api/ai/predictions        — Prophet fault volume forecast
    GET  /api/ai/clusters           — K-Means geographic clusters
    GET  /api/ai/resource-plan      — SRS 5.6.8: Predictive Resource Planning (FR-33)
    POST /api/ai/shortest-path      — SRS 5.6.6: Dijkstra technician-to-fault route (FR-29)
    GET  /api/ai/dashboard          — Combined dashboard data
    POST /api/ai/train                       — SRS 5.6.7: CSV upload + async retrain (both models)
    GET  /api/ai/train/status/{jobId}        — Poll async training job progress

Run:
    python app.py                    (development)
    gunicorn -w 2 -b 0.0.0.0:5000 app:app   (production)
"""

import os
import logging
import threading
from datetime import datetime, timedelta
from typing import Any

from flask import Flask, jsonify, request
from flask_cors import CORS

from config import Config, logger, is_db_available
from utils.formatters import _sanitise
from utils.validators import validate_shortest_path_request

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

try:
    from models.resource_planner import ResourcePlanner
    _has_resource_planner = True
except ImportError as e:
    logger.warning(f"ResourcePlanner not available: {e}")
    _has_resource_planner = False

# ─── Data layer ───────────────────────────────────────────────────────────────
from data.data_extractor import DataExtractor
from data.synthetic_data import SyntheticDataGenerator
from data.upload_manager import UploadManager
from data.training_job_store import TrainingJobStore

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
_uploads    = UploadManager()
_forecaster = ProphetForecaster()   if _has_forecaster else None
_clusterer  = KMeansClusterer()     if _has_clusterer  else None
_router     = DijkstraRouter()      if _has_router     else None
_jobs       = TrainingJobStore()

# ResourcePlanner is a pure combiner over the forecaster/clusterer/extractor
# above — only meaningful once both underlying models are available.
_has_resource_planner = _has_resource_planner and _has_forecaster and _has_clusterer
_resource_planner = (
    ResourcePlanner(_forecaster, _clusterer, _extractor)
    if _has_resource_planner else None
)


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def ok(data: Any, message: str = "OK", status: int = 200) -> tuple:
    # _sanitise recursively converts NaN/Inf -> None and numpy/pandas types
    # to native Python types. Without this, a NaN slipping in from a SQL
    # LEFT JOIN (e.g. a technician with no current job) gets serialised as
    # a bare `NaN` token by Flask's jsonify, which is not valid JSON and
    # breaks JSON.parse() on the frontend.
    return jsonify({
        "success":   True,
        "data":      _sanitise(data),
        "message":   message,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }), status


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
            "resource_planner":   "available" if _has_resource_planner else "not loaded",
        },
        "config": {
            "forecast_horizon_days":    Config.FORECAST_HORIZON_DAYS,
            "kmeans_clusters":          Config.KMEANS_N_CLUSTERS,
            "route_radius_km":          Config.ROUTE_SEARCH_RADIUS_KM,
            "route_avg_speed_kmh":      Config.ROUTE_AVG_SPEED_KMH,
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

    Returns (normal):
        {
          historical: [{ds, y, actual}],
          forecast:   [{ds, yhat, upper, lower, isForecast}],
          metrics:    {mae, rmse, accuracy, trainingDays, horizon, lastTrained},
          dataSource: "database" | "synthetic"
        }

    Returns (SRS 5.6.2 — real DB data exists but doesn't meet the 6-month
    minimum; historical/forecast are empty rather than a substituted
    synthetic-data forecast dressed up as real):
        {
          insufficientData: true,
          historyDays:  int,
          requiredDays: int,
          historical: [], forecast: [], metrics: {...all null...},
          dataSource: "database"
        }
    """
    if not _has_forecaster or _forecaster is None:
        return err("Prophet forecasting model not available", 503)

    horizon = _int_param('horizon', Config.FORECAST_HORIZON_DAYS, 7, 90)

    try:
        if is_db_available():
            raw_df = _extractor.get_fault_time_series(
                days_back=max(Config.FORECAST_MIN_HISTORY_DAYS * 2, 540)
            )
            data_source = 'database'
        else:
            raw_df = None
            data_source = 'synthetic'

        # Fall back to synthetic ONLY when there's no DB connection at all —
        # NOT when the DB has real data that's merely thinner than the
        # 6-month minimum. That distinction is what changed here: real-but-
        # insufficient data used to be silently discarded in favour of a
        # full synthetic substitute; now it's passed through honestly and
        # forecast() itself reports `insufficientData` (see below).
        if raw_df is None:
            logger.info("Using synthetic data for forecast (DB unavailable)")
            raw_df = _synth.fault_time_series(days=540)
            data_source = 'synthetic'

        result = _forecaster.forecast(raw_df, horizon=horizon)
        result['dataSource'] = data_source

        if result.get('insufficientData'):
            logger.info(
                f"Insufficient historical data for forecast: "
                f"{result.get('historyDays')} / {result.get('requiredDays')} days"
            )
            return ok(result, "Insufficient historical data for a reliable forecast")

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


@app.route('/api/ai/resource-plan', methods=['GET'])
def resource_plan():
    """
    SRS 5.6.8 — Predictive Resource Planning (FR-33). Combines the fault
    volume forecast (5.6.2), geographic demand clusters (5.6.3), and
    historical shift/material ratios into predicted hotspots (date + shift
    window + zone) with suggested Technician/Vehicle/Material quantities.
    Advisory only — this endpoint does not write anything to any BOD
    dispatch screen (that pre-population is a separate, later Admin UI
    concern).

    Query params:
        horizon (int): Days ahead to plan for (default 7, range 7–90 —
                       same range as /api/ai/predictions' horizon param).

    Returns (insufficient data — SRS 5.6.2/5.6.8's shared 6-month rule):
        {insufficientData: true, historyDays, requiredDays,
         hotspots: [], materialShortfalls: []}

    Returns (normal):
        {
          insufficientData: false,
          horizonDays: int,
          hotspots: [{date, shift, zoneId, zoneName, predictedFaultCount,
                      suggestedTechnicians, suggestedVehicles,
                      materials: [{materialId, materialName, suggestedQuantity}]}],
          materialShortfalls: [{materialId, materialName, currentStock,
                                 totalSuggestedQuantity, insufficient}],
          shiftDistributionSource: 'historical' | 'even_split_fallback',
              # 'even_split_fallback' means there's no historical fault-shift
              # data yet (e.g. a brand-new deployment) and every hotspot's
              # shift split was assumed even 1/3 rather than measured --
              # same purpose as insufficientData above, but for the shift
              # half of the plan specifically rather than the whole forecast.
          generatedAt: str
        }
    """
    if not _has_resource_planner or _resource_planner is None:
        return err("Resource planner not available (requires both the forecaster and clusterer)", 503)

    horizon = _int_param('horizon', 7, 7, 90)

    try:
        # Fault time-series for the forecast half — same fetch-vs-synthetic
        # pattern as /api/ai/predictions. Real-but-thin DB data is passed
        # through honestly; ProphetForecaster.forecast() itself reports
        # insufficientData rather than this handler pre-guessing.
        if is_db_available():
            raw_df = _extractor.get_fault_time_series(
                days_back=max(Config.FORECAST_MIN_HISTORY_DAYS * 2, 540)
            )
        else:
            raw_df = None
        if raw_df is None:
            raw_df = _synth.fault_time_series(days=540)

        # GPS points for the zone half — same fetch-vs-synthetic pattern as
        # /api/ai/clusters.
        gps_df = None
        if is_db_available():
            gps_df = _extractor.get_faults_with_location(days_back=180)
            if gps_df is None or len(gps_df) < 20:
                gps_df = None
        if gps_df is None:
            gps_df = _synth.fault_gps_points(n=1000, days_back=180)

        result = _resource_planner.generate_plan(raw_df, gps_df, horizon_days=horizon)

        if result.get('insufficientData'):
            return ok(result, "Insufficient historical data for a reliable resource plan")

        return ok(result, f"{horizon}-day predicted resource plan: {len(result['hotspots'])} hotspots")

    except Exception as exc:
        logger.exception("Resource plan error")
        return err(f"Resource plan failed: {str(exc)}")


@app.route('/api/ai/shortest-path', methods=['POST'])
def shortest_path():
    """
    SRS 5.6.6 — Technician-to-fault shortest-path navigation (FR-29).

    Computes the shortest path from a Technician's current live location
    to their assigned fault's location via Dijkstra's algorithm, with a
    Haversine straight-line fallback where road-network graph data is
    unavailable. No road-graph data source is wired up yet, so every
    call currently uses the fallback — see models/route_optimizer.py.

    Stage 1: AI/ML backend contract only. Mobile-side rendering is Stage 2.

    POST body: { currentLat, currentLng, faultLat, faultLng }

    Returns:
        {
          waypoints:   [{lat, lng}, ...] ordered start -> end,
          distanceKm:  float,
          etaMinutes:  int,
          routed:      bool (false = Haversine straight-line fallback),
          algorithm:   str,
          avgSpeedKmh: float
        }
    """
    if not _has_router or _router is None:
        return err("Dijkstra router not available", 503)

    body = request.get_json(silent=True) or {}
    validated, verr = validate_shortest_path_request(body)
    if verr:
        return err(verr, 400)

    try:
        result = _router.route(
            validated['currentLat'], validated['currentLng'],
            validated['faultLat'],   validated['faultLng'],
        )
        message = (
            "Shortest path computed"
            if result['routed']
            else "Shortest path computed (Haversine straight-line fallback — no road-network data source configured)"
        )
        return ok(result, message)
    except Exception as exc:
        logger.exception("Shortest-path error")
        return err(f"Shortest-path computation failed: {str(exc)}")


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
                    forecast_full = _forecaster.forecast(raw_df, horizon=7)
                    if forecast_full.get('insufficientData'):
                        # SRS 5.6.2 — same honesty rule as /api/ai/predictions:
                        # real-but-thin DB data reports insufficiency rather
                        # than a forecast_summary built from a silently-
                        # substituted synthetic forecast.
                        forecast_summary = {
                            'insufficientData': True,
                            'historyDays':      forecast_full.get('historyDays'),
                            'requiredDays':     forecast_full.get('requiredDays'),
                        }
                    else:
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
                'forecaster':      _has_forecaster,
                'clusterer':       _has_clusterer,
                'router':          _has_router,
                'resourcePlanner': _has_resource_planner,
            },
            'generatedAt': datetime.utcnow().isoformat() + "Z",
        }
        return ok(payload, "Dashboard data loaded")

    except Exception as exc:
        logger.exception("Dashboard error")
        return err(f"Dashboard failed: {str(exc)}")


def _run_training_job(job_id: str, raw_df) -> None:
    """
    Background thread target for POST /api/ai/train. Retrains both the
    forecaster and clusterer from the uploaded CSV (SRS 5.6.7: one upload
    retrains Prophet AND refreshes K-Means), via Stage B's retrain() —
    which produces a CANDIDATE + comparison and does NOT auto-activate.
    Progress and the final result are recorded in the job store so
    GET /api/ai/train/status/{jobId} can poll them from any worker process.
    """
    try:
        _jobs.update_status(job_id, 'preprocessing')
        ts_df   = _uploads.to_time_series(raw_df)
        gps_df  = _uploads.to_gps_points(raw_df)
        area_df = _uploads.to_exchange_area_groups(raw_df)

        _jobs.update_status(job_id, 'training')
        result = {}

        if _has_forecaster and _forecaster is not None:
            try:
                fc_result = _forecaster.retrain(ts_df)
                result['forecaster'] = (
                    {'status': 'error', 'message': fc_result['error']}
                    if 'error' in fc_result else fc_result
                )
            except Exception as exc:
                result['forecaster'] = {'status': 'error', 'message': str(exc)}

        if _has_clusterer and _clusterer is not None:
            has_gps  = gps_df is not None and len(gps_df) >= 10
            has_area = not has_gps and area_df is not None and len(area_df) >= 1

            if has_gps:
                # Real coordinates win when both are present — strictly more
                # informative than a categorical code, and this is the existing,
                # versioned/Activate-able K-Means path (SRS 5.6.7 governance).
                try:
                    cl_result = _clusterer.retrain(gps_df)
                    result['clusterer'] = (
                        {'status': 'error', 'message': cl_result['error']}
                        if 'error' in cl_result else cl_result
                    )
                except Exception as exc:
                    result['clusterer'] = {'status': 'error', 'message': str(exc)}

            elif has_area:
                # Categorical fallback (H3) — no GPS in this upload (the real WFMS
                # export shape: EXCHANGEAREA codes, no lat/lng). This is a
                # deterministic groupby, not a fitted model, so it has no
                # versionId/candidate to Activate — the existing ModelResultCard
                # UI only knows 'status: error' (shows the message) or a
                # candidate-with-versionId shape (renders an Activate button
                # that would 400 here, since no version exists). Reporting it
                # as 'error' is the honest choice available today, not a real
                # failure — the actual computed zones are attached under
                # 'categoricalPreview' so the data isn't silently discarded,
                # inspectable via GET /api/ai/train/status/{jobId}. A dedicated
                # frontend treatment (render the zones, no Activate button) is
                # a natural follow-up, out of scope for this backend change.
                try:
                    area_result = _clusterer.cluster_by_exchange_area(area_df)
                    result['clusterer'] = {
                        'status': 'error',
                        'message': (
                            f"No GPS columns in this upload — grouped by EXCHANGEAREA "
                            f"instead ({area_result['nClusters']} zone(s), "
                            f"{area_result['totalFaults']} fault(s)). This is categorical "
                            f"grouping, not a fitted K-Means model, so there is no version "
                            f"to Activate. See categoricalPreview for the computed zones."
                        ),
                        'categoricalPreview': area_result,
                    }
                except Exception as exc:
                    result['clusterer'] = {'status': 'error', 'message': str(exc)}

            else:
                result['clusterer'] = {
                    'status': 'error',
                    'message': 'Uploaded data has no usable latitude/longitude columns, '
                               'and no usable EXCHANGEAREA column either — nothing to '
                               'cluster on.',
                }

        _jobs.set_result(job_id, result)
    except Exception as exc:
        logger.exception(f"Training job {job_id} crashed")
        _jobs.set_error(job_id, str(exc))


@app.route('/api/ai/train', methods=['POST'])
def train():
    """
    SRS 5.6.7 — CSV upload + retrain (async, the spec'd endpoint).

    Validates the CSV (Stage A: 24-month cap, row cap, date/GPS
    validation — see data/upload_manager.py), then retrains BOTH models
    in a background thread and returns a jobId immediately — training
    does not block the request.
    Poll GET /api/ai/train/status/{jobId} for progress and, once
    complete, the candidate + comparison result for each model. Nothing
    auto-activates (Stage B); call
    POST /api/ai/model-versions/<model>/activate separately to promote.

    Form data: file (CSV)
    """
    if 'file' not in request.files:
        return err("No file uploaded. Send it as multipart/form-data field 'file'.", 400)

    try:
        # save_upload() returns the validated DataFrame from the SAME call
        # that parsed it — no separate disk read-back afterwards, so a
        # second admin's concurrent upload has no gap to land in before
        # this job's data is captured (see its docstring).
        summary, raw_df = _uploads.save_upload(request.files['file'])
    except ValueError as exc:
        return err(str(exc), 400)
    except Exception as exc:
        logger.exception("Upload error")
        return err(f"Upload failed: {str(exc)}")

    job_id = _jobs.create_job(summary)
    thread = threading.Thread(target=_run_training_job, args=(job_id, raw_df.copy()), daemon=True)
    thread.start()

    return ok(
        {'jobId': job_id, 'status': 'queued', 'uploadSummary': summary},
        "Training job queued",
        status=202,
    )


@app.route('/api/ai/train/status/<job_id>', methods=['GET'])
def train_status(job_id):
    """Poll async training job progress (SRS 5.6.7)."""
    job = _jobs.get(job_id)
    if job is None:
        return err(f"Unknown job '{job_id}'.", 404)
    return ok(job, f"Job {job_id}: {job['status']}")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL VERSIONING — SRS 5.6.7 governance (Activate Model / rollback)
#
# Synchronous, admin-triggered actions on a job's resulting candidate(s).
# Not themselves spec'd endpoints, but the only way to promote a candidate
# produced by POST /api/ai/train (or the legacy upload+preview flow below).
# ─────────────────────────────────────────────────────────────────────────────

def _model_by_name(name: str):
    if name == 'forecaster' and _has_forecaster and _forecaster is not None:
        return _forecaster
    if name == 'clusterer' and _has_clusterer and _clusterer is not None:
        return _clusterer
    return None


@app.route('/api/ai/model-versions/<model_name>', methods=['GET'])
def list_model_versions(model_name):
    """List all known versions (candidate/active/archived) for a model."""
    model = _model_by_name(model_name)
    if model is None:
        return err(f"Unknown or unavailable model '{model_name}'. Use 'forecaster' or 'clusterer'.", 400)
    return ok({'versions': model.list_versions()}, f"Versions for {model_name}")


@app.route('/api/ai/model-versions/<model_name>/activate', methods=['POST'])
def activate_model_version(model_name):
    """
    Explicit 'Activate Model' action (SRS 5.6.7) — promotes a candidate (or
    any archived version) to be the model that serves live requests.
    POST body: { "versionId": <int> }
    """
    model = _model_by_name(model_name)
    if model is None:
        return err(f"Unknown or unavailable model '{model_name}'. Use 'forecaster' or 'clusterer'.", 400)

    body = request.get_json(silent=True) or {}
    version_id = body.get('versionId')
    if version_id is None:
        return err("Missing required field 'versionId'.", 400)

    try:
        entry = model.activate_version(int(version_id))
        return ok(entry, f"Version {version_id} activated for {model_name}")
    except ValueError as exc:
        return err(str(exc), 404)
    except Exception as exc:
        logger.exception("Activate error")
        return err(f"Activate failed: {str(exc)}")


@app.route('/api/ai/model-versions/<model_name>/rollback', methods=['POST'])
def rollback_model_version(model_name):
    """
    Revert the active model. POST body: { "versionId": <int> } (optional —
    defaults to the version that was active immediately before the current one).
    """
    model = _model_by_name(model_name)
    if model is None:
        return err(f"Unknown or unavailable model '{model_name}'. Use 'forecaster' or 'clusterer'.", 400)

    body = request.get_json(silent=True) or {}
    version_id = body.get('versionId')

    try:
        entry = model.rollback(int(version_id) if version_id is not None else None)
        return ok(entry, f"Rolled back to version {entry['versionId']} for {model_name}")
    except ValueError as exc:
        return err(str(exc), 404)
    except Exception as exc:
        logger.exception("Rollback error")
        return err(f"Rollback failed: {str(exc)}")


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
