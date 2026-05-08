"""
routes/clusters.py — /api/ai/clusters Blueprint
================================================
K-Means geographic demand clustering endpoint.

Endpoints:
    GET  /api/ai/clusters                Default k=5 clusters
    GET  /api/ai/clusters?n_clusters=3   Custom cluster count
    GET  /api/ai/clusters/map            Cluster data formatted for SVG map
    GET  /api/ai/clusters/heatmap        Grid-density heatmap data
    POST /api/ai/clusters/retrain        Force K-Means refit

Query parameters:
    n_clusters (int): Number of clusters. Default 5, range 2–10.
    days       (int): History window in days. Default 180, max 730.
    category   (str): Filter faults to one category before clustering.

Response shape (GET /clusters):
    {
      success: bool,
      data: {
        clusters: [{
          clusterId, rank, regionName, faultCount, faultPercent,
          riskLevel, riskScore, topCategory,
          centroid: { lat, lng },
          density, techniciansNeeded, color
        }],
        totalFaults:      int,
        silhouetteScore:  float | null,
        nClusters:        int,
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
from data.feature_engineer import haversine_km

logger = logging.getLogger('slt_ai.routes.clusters')

clusters_bp = Blueprint('clusters', __name__)

# ── Singleton ──────────────────────────────────────────────────────────────────
_clusterer = None

def _get_clusterer():
    global _clusterer
    if _clusterer is None:
        try:
            from models.clustering import KMeansClusterer
            _clusterer = KMeansClusterer()
        except Exception as e:
            logger.error(f"Could not load KMeansClusterer: {e}")
    return _clusterer


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _ok(data, message='OK', code=200):
    return jsonify({
        'success':   True,
        'data':      data,
        'message':   message,
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }), code


def _err(message, code=500):
    logger.warning(f"Clusters error [{code}]: {message}")
    return jsonify({
        'success':   False,
        'error':     message,
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }), code


def _parse_n_clusters() -> int:
    try:
        n = int(request.args.get('n_clusters', Config.KMEANS_N_CLUSTERS))
        return max(2, min(10, n))
    except (ValueError, TypeError):
        return Config.KMEANS_N_CLUSTERS


def _parse_days() -> int:
    try:
        d = int(request.args.get('days', 180))
        return max(30, min(730, d))
    except (ValueError, TypeError):
        return 180


def _load_gps_data(days: int, category: str = None):
    """
    Load fault GPS data from MySQL or synthetic fallback.
    Returns (DataFrame, data_source_str).
    """
    from data.data_extractor import DataExtractor
    from data.synthetic_data import SyntheticDataGenerator

    if is_db_available():
        extractor     = DataExtractor()
        status_filter = None  # include all statuses for clustering
        if category:
            # Post-filter by category — extractor returns all, we filter
            gps_df = extractor.get_faults_with_location(days_back=days)
            if gps_df is not None and not gps_df.empty:
                gps_df = gps_df[gps_df['category'] == category.upper()]
                if len(gps_df) >= 20:
                    return gps_df, 'database'
        else:
            gps_df = extractor.get_faults_with_location(days_back=days)
            if gps_df is not None and len(gps_df) >= 20:
                return gps_df, 'database'

    logger.info("Clusters: using synthetic GPS data")
    synth  = SyntheticDataGenerator(seed=42)
    gps_df = synth.fault_gps_points(n=1000, days_back=days)
    if category:
        gps_df = gps_df[gps_df['category'] == category.upper()]
    return gps_df, 'synthetic'


# ─── Routes ───────────────────────────────────────────────────────────────────

@clusters_bp.route('/clusters', methods=['GET'])
def get_clusters():
    """
    Main clustering endpoint.

    GET /api/ai/clusters
    GET /api/ai/clusters?n_clusters=3&days=90
    GET /api/ai/clusters?category=FIBER
    """
    clusterer = _get_clusterer()
    if clusterer is None:
        return _err('K-Means clustering model not available', 503)

    n_clusters = _parse_n_clusters()
    days       = _parse_days()
    category   = request.args.get('category', '').upper() or None

    if category and category not in Config.FAULT_CATEGORIES + ['ALL', '']:
        return _err(
            f"Invalid category '{category}'. Valid: {', '.join(Config.FAULT_CATEGORIES)}",
            400
        )

    try:
        gps_df, data_source = _load_gps_data(
            days,
            category if category and category != 'ALL' else None
        )

        result = clusterer.cluster(gps_df, n_clusters=n_clusters)
        result['dataSource'] = data_source

        if category:
            result['filteredCategory'] = category

        logger.info(
            f"Clusters served: k={n_clusters}, "
            f"points={result.get('totalFaults')}, "
            f"source={data_source}"
        )
        return _ok(result, f"K-Means clustering: {n_clusters} clusters over {days} days")

    except Exception as exc:
        logger.exception("Clustering error")
        return _err(f"Clustering failed: {str(exc)}")


@clusters_bp.route('/clusters/map', methods=['GET'])
def get_clusters_for_map():
    """
    Cluster data formatted for the SVG Sri Lanka map in the admin portal.

    Returns centroid positions in screen-space (0–560 x 0–300)
    in addition to raw lat/lng, so the frontend doesn't need to
    do the projection math.

    GET /api/ai/clusters/map
    """
    clusterer = _get_clusterer()
    if clusterer is None:
        return _err('Clustering model not available', 503)

    n_clusters = _parse_n_clusters()
    days       = _parse_days()

    # SVG viewport dimensions (must match AIDashboardPage.js)
    SVG_W, SVG_H = 560, 300

    try:
        gps_df, data_source = _load_gps_data(days)
        result = clusterer.cluster(gps_df, n_clusters=n_clusters)

        map_clusters = []
        for c in result.get('clusters', []):
            lat = c['centroid']['lat']
            lng = c['centroid']['lng']

            # Project lat/lng to SVG pixel coordinates
            px = ((lng - Config.SL_LNG_MIN) / (Config.SL_LNG_MAX - Config.SL_LNG_MIN)) * SVG_W
            py = (1 - (lat - Config.SL_LAT_MIN) / (Config.SL_LAT_MAX - Config.SL_LAT_MIN)) * SVG_H
            # Bubble radius: proportional to fault count (min 12, max 38)
            radius = max(12, min(38, c['faultCount'] * 0.8))

            map_clusters.append({
                **c,
                'svgX':    round(px, 1),
                'svgY':    round(py, 1),
                'radius':  round(radius, 1),
            })

        return _ok(
            {
                'clusters':       map_clusters,
                'totalFaults':    result.get('totalFaults'),
                'silhouetteScore': result.get('silhouetteScore'),
                'dataSource':     data_source,
                'svgViewBox':     f"0 0 {SVG_W} {SVG_H}",
            },
            "Map-ready cluster data"
        )

    except Exception as exc:
        logger.exception("Map clusters error")
        return _err(f"Map cluster failed: {str(exc)}")


@clusters_bp.route('/clusters/heatmap', methods=['GET'])
def get_heatmap():
    """
    Grid-density heatmap for the admin live map (Leaflet heatmap layer).

    Returns a flat list of { lat, lng, intensity } points where
    intensity is normalised 0–1 based on local fault density.

    GET /api/ai/clusters/heatmap?days=90&resolution=50
    """
    days       = _parse_days()
    resolution = max(10, min(200, int(request.args.get('resolution', 50))))

    try:
        gps_df, data_source = _load_gps_data(days)
        if gps_df is None or gps_df.empty:
            return _ok({'points': [], 'dataSource': 'none'}, 'No data')

        # Build grid
        import numpy as np
        lat_bins = np.linspace(Config.SL_LAT_MIN, Config.SL_LAT_MAX, resolution)
        lng_bins = np.linspace(Config.SL_LNG_MIN, Config.SL_LNG_MAX, resolution)

        # Count faults per grid cell
        lat_idx = np.digitize(gps_df['latitude'].values,  lat_bins) - 1
        lng_idx = np.digitize(gps_df['longitude'].values, lng_bins) - 1
        grid    = np.zeros((resolution, resolution), dtype=float)

        for la, lo in zip(lat_idx, lng_idx):
            if 0 <= la < resolution and 0 <= lo < resolution:
                grid[la, lo] += 1

        # Normalise and flatten (skip empty cells)
        max_val = grid.max()
        if max_val == 0:
            return _ok({'points': [], 'dataSource': data_source}, 'No density data')

        points = []
        for la in range(resolution):
            for lo in range(resolution):
                if grid[la, lo] > 0:
                    points.append({
                        'lat':       round(float(lat_bins[la]), 4),
                        'lng':       round(float(lng_bins[lo]), 4),
                        'intensity': round(float(grid[la, lo] / max_val), 3),
                        'count':     int(grid[la, lo]),
                    })

        return _ok(
            {'points': points, 'totalPoints': len(points), 'dataSource': data_source},
            f"Heatmap grid ({resolution}×{resolution})"
        )

    except Exception as exc:
        logger.exception("Heatmap error")
        return _err(f"Heatmap failed: {str(exc)}")


@clusters_bp.route('/clusters/retrain', methods=['POST'])
def retrain_clusters():
    """
    Force K-Means refit on latest data.

    POST /api/ai/clusters/retrain
    Body: { "n_clusters": 5, "days": 180 }
    """
    clusterer = _get_clusterer()
    if clusterer is None:
        return _err('Clustering model not available', 503)

    body       = request.get_json(silent=True) or {}
    n_clusters = max(2, min(10, int(body.get('n_clusters', Config.KMEANS_N_CLUSTERS))))
    days       = max(30, min(730, int(body.get('days', 180))))

    try:
        gps_df, data_source = _load_gps_data(days)
        info = clusterer.retrain(gps_df)
        info['dataSource'] = data_source

        logger.info(f"K-Means retrained: k={n_clusters}, points={info.get('trainingPoints')}")
        return _ok(info, "K-Means model refitted successfully")

    except Exception as exc:
        logger.exception("Cluster retrain error")
        return _err(f"Retrain failed: {str(exc)}")
