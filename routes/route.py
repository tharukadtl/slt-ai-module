"""
routes/route.py — /api/ai/optimize-route Blueprint
===================================================
Dijkstra + Haversine nearest-technician route optimisation endpoint.

Endpoints:
    GET  /api/ai/optimize-route          Find nearest technicians
    POST /api/ai/optimize-route/batch    Batch multi-fault assignment
    GET  /api/ai/optimize-route/nearby   Nearby technicians (no Dijkstra)

Query parameters (GET /optimize-route):
    lat           (float): Fault latitude  — REQUIRED
    lng           (float): Fault longitude — REQUIRED
    limit         (int):   Max results to return. Default 5, max 20.
    available_only (bool): Only return AVAILABLE techs. Default false.

Response shape (GET /optimize-route):
    {
      success: bool,
      data: {
        faultLocation:  { lat, lng },
        technicians: [{
          rank, technicianId, technicianName, phone, branchName,
          latitude, longitude,
          distanceKm, weightedDistanceKm, estimatedArrivalMinutes,
          status, isAvailable, currentJobId, avatarInitial
        }],
        algorithm:        "dijkstra+haversine",
        totalCandidates:  int,
        searchRadiusKm:   float,
        dataSource:       "database" | "synthetic",
      },
      message:   str,
      timestamp: ISO8601
    }
"""

import logging
from datetime import datetime, timezone

from flask import Blueprint, jsonify, request

from config import Config, is_db_available
from data.feature_engineer import haversine_km, estimate_travel_minutes

logger = logging.getLogger('slt_ai.routes.route')

route_bp = Blueprint('route', __name__)

# ── Singleton ──────────────────────────────────────────────────────────────────
_router = None

def _get_router():
    global _router
    if _router is None:
        try:
            from models.route_optimizer import DijkstraRouter
            _router = DijkstraRouter()
        except Exception as e:
            logger.error(f"Could not load DijkstraRouter: {e}")
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
    logger.warning(f"Route error [{code}]: {message}")
    return jsonify({
        'success':   False,
        'error':     message,
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }), code


def _parse_coords() -> tuple:
    """Parse and validate lat/lng from query params. Returns (lat, lng) or raises ValueError."""
    try:
        lat = float(request.args.get('lat', 0))
        lng = float(request.args.get('lng', 0))
    except (ValueError, TypeError):
        raise ValueError("lat and lng must be valid numbers")

    if not (Config.SL_LAT_MIN <= lat <= Config.SL_LAT_MAX):
        raise ValueError(
            f"Latitude {lat} out of Sri Lanka bounds "
            f"[{Config.SL_LAT_MIN}, {Config.SL_LAT_MAX}]"
        )
    if not (Config.SL_LNG_MIN <= lng <= Config.SL_LNG_MAX):
        raise ValueError(
            f"Longitude {lng} out of Sri Lanka bounds "
            f"[{Config.SL_LNG_MIN}, {Config.SL_LNG_MAX}]"
        )
    return lat, lng


def _load_technicians():
    """
    Load live technician locations from MySQL or synthetic fallback.
    Returns (DataFrame, data_source_str).
    """
    from data.data_extractor import DataExtractor
    from data.synthetic_data import SyntheticDataGenerator

    if is_db_available():
        extractor = DataExtractor()
        tech_df   = extractor.get_available_technicians()
        if tech_df is not None and not tech_df.empty:
            return tech_df, 'database'

        # Try all technicians if no recent GPS
        all_df = extractor.get_all_technicians()
        if all_df is not None and not all_df.empty:
            logger.warning("No recent GPS data — using all technicians without location filter")
            return all_df, 'database_no_gps'

    logger.info("Route: using synthetic technician data")
    synth = SyntheticDataGenerator(seed=42)
    return synth.technician_locations(n=25), 'synthetic'


# ─── Routes ───────────────────────────────────────────────────────────────────

@route_bp.route('/optimize-route', methods=['GET'])
def optimize_route():
    """
    Main route optimisation endpoint.

    GET /api/ai/optimize-route?lat=6.9271&lng=79.8612
    GET /api/ai/optimize-route?lat=6.9271&lng=79.8612&limit=3&available_only=true
    """
    router = _get_router()
    if router is None:
        return _err('Route optimisation model not available', 503)

    try:
        fault_lat, fault_lng = _parse_coords()
    except ValueError as ve:
        return _err(str(ve), 400)

    limit          = max(1, min(20, int(request.args.get('limit', 5))))
    available_only = request.args.get('available_only', 'false').lower() == 'true'

    try:
        tech_df, data_source = _load_technicians()

        result = router.find_nearest(
            technicians_df=tech_df,
            fault_lat=fault_lat,
            fault_lng=fault_lng,
            limit=limit,
            available_only=available_only,
        )
        result['faultLocation'] = {'lat': fault_lat, 'lng': fault_lng}
        result['dataSource']    = data_source

        n_found = len(result.get('technicians', []))
        logger.info(
            f"Route optimised: fault=({fault_lat:.4f},{fault_lng:.4f}), "
            f"found={n_found}, source={data_source}"
        )
        return _ok(result, f"Found {n_found} nearest technician(s)")

    except Exception as exc:
        logger.exception("Route optimisation error")
        return _err(f"Route optimisation failed: {str(exc)}")


@route_bp.route('/optimize-route/batch', methods=['POST'])
def batch_assign():
    """
    Greedy batch assignment for multiple faults.

    Assigns the nearest available technician to each fault in priority order,
    preventing the same technician from being assigned to multiple faults.

    POST /api/ai/optimize-route/batch
    Body: {
      "faults": [
        { "fault_id": 1, "lat": 6.93, "lng": 79.86, "priority": "HIGH" },
        { "fault_id": 2, "lat": 7.29, "lng": 80.63, "priority": "MEDIUM" }
      ]
    }

    Response: {
      "assignments": [
        { "fault_id": 1, "technician_id": 5, "distance_km": 2.3, "eta_minutes": 4 },
        ...
      ],
      "totalAssigned": int,
      "totalFailed":   int
    }
    """
    router = _get_router()
    if router is None:
        return _err('Route optimisation model not available', 503)

    body   = request.get_json(silent=True)
    if not body or 'faults' not in body:
        return _err("Request body must contain 'faults' array", 400)

    faults = body['faults']
    if not isinstance(faults, list) or len(faults) == 0:
        return _err("'faults' must be a non-empty array", 400)
    if len(faults) > 50:
        return _err("Maximum 50 faults per batch request", 400)

    # Validate each fault entry
    valid_faults  = []
    invalid_items = []
    priority_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}

    for i, f in enumerate(faults):
        try:
            lat = float(f.get('lat', 0))
            lng = float(f.get('lng', 0))
            if not (Config.SL_LAT_MIN <= lat <= Config.SL_LAT_MAX and
                    Config.SL_LNG_MIN <= lng <= Config.SL_LNG_MAX):
                invalid_items.append({'index': i, 'error': 'Coordinates out of Sri Lanka bounds'})
                continue
            valid_faults.append({
                'fault_id': f.get('fault_id', i + 1),
                'lat':      lat,
                'lng':      lng,
                'priority': str(f.get('priority', 'MEDIUM')).upper(),
            })
        except (ValueError, TypeError):
            invalid_items.append({'index': i, 'error': 'Invalid lat/lng'})

    if not valid_faults:
        return _err("No valid faults after validation", 400)

    # Sort by priority (HIGH first)
    valid_faults.sort(key=lambda f: priority_order.get(f['priority'], 1))

    try:
        tech_df, data_source = _load_technicians()
        assignments = router.batch_assign(tech_df, valid_faults)

        total_assigned = sum(1 for a in assignments if 'technician_id' in a)
        total_failed   = len(assignments) - total_assigned

        return _ok({
            'assignments':   assignments,
            'totalAssigned': total_assigned,
            'totalFailed':   total_failed,
            'invalidInputs': invalid_items,
            'dataSource':    data_source,
        }, f"Batch assignment: {total_assigned}/{len(faults)} faults assigned")

    except Exception as exc:
        logger.exception("Batch assignment error")
        return _err(f"Batch assignment failed: {str(exc)}")


@route_bp.route('/optimize-route/nearby', methods=['GET'])
def nearby_technicians():
    """
    Simple radius-based nearby technician lookup (no Dijkstra).
    Faster than the full optimiser — used for quick proximity checks.

    GET /api/ai/optimize-route/nearby?lat=6.93&lng=79.86&radius=20

    Query params:
        lat    (float): Fault latitude
        lng    (float): Fault longitude
        radius (float): Search radius in km (default 20, max 100)
    """
    try:
        fault_lat, fault_lng = _parse_coords()
    except ValueError as ve:
        return _err(str(ve), 400)

    radius = max(1.0, min(100.0, float(request.args.get('radius', 20.0))))

    try:
        tech_df, data_source = _load_technicians()

        if tech_df is None or tech_df.empty:
            return _ok({
                'technicians': [], 'count': 0, 'radiusKm': radius,
                'faultLocation': {'lat': fault_lat, 'lng': fault_lng},
                'dataSource': data_source,
            }, 'No technicians available')

        # Compute distances for all technicians
        results = []
        for _, row in tech_df.iterrows():
            t_lat = float(row.get('latitude', 0) or 0)
            t_lng = float(row.get('longitude', 0) or 0)
            if not t_lat or not t_lng:
                continue

            dist = haversine_km(fault_lat, fault_lng, t_lat, t_lng)
            if dist <= radius:
                eta = estimate_travel_minutes(dist)
                results.append({
                    'technicianId':            int(row.get('technician_id', 0)),
                    'technicianName':          str(row.get('full_name', 'Unknown')),
                    'phone':                   str(row.get('phone', '')),
                    'distanceKm':              round(dist, 2),
                    'estimatedArrivalMinutes': eta,
                    'status':                  str(row.get('status', 'UNKNOWN')),
                    'latitude':                round(t_lat, 6),
                    'longitude':               round(t_lng, 6),
                    'isAvailable':             str(row.get('status', '')) in Config.TECH_AVAILABLE_STATUSES,
                })

        # Sort by distance
        results.sort(key=lambda r: r['distanceKm'])

        return _ok({
            'technicians':  results,
            'count':        len(results),
            'radiusKm':     radius,
            'faultLocation': {'lat': fault_lat, 'lng': fault_lng},
            'dataSource':   data_source,
        }, f"{len(results)} technician(s) within {radius}km")

    except Exception as exc:
        logger.exception("Nearby technicians error")
        return _err(f"Nearby lookup failed: {str(exc)}")
