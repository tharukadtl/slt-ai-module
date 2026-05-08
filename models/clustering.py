"""
models/clustering.py — K-Means Geographic Fault Clustering
============================================================
Groups fault GPS coordinates into k=5 geographic demand zones
across Sri Lanka using scikit-learn K-Means.

SRS requirement:
  - k=5 clusters (configurable)
  - Identify high-demand geographic zones for technician pre-positioning
  - Assign risk level (HIGH / MEDIUM / LOW) based on fault density
  - Label clusters with nearest Sri Lanka district names

Pipeline:
  1. DataExtractor / SyntheticGenerator supplies GPS DataFrame
  2. DataCleaner validates coordinates and clips to SL bounds
  3. FeatureEngineer enriches with distance-from-Colombo, region flags
  4. KMeansClusterer fits, labels, computes risk, returns API-ready dict

Usage:
    from models.clustering import KMeansClusterer
    kc = KMeansClusterer()
    result = kc.cluster(gps_df, n_clusters=5)
    # result keys: clusters, totalFaults, silhouetteScore
"""

import logging
import math
import pickle
import pathlib
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd

from config import Config
from data.feature_engineer import haversine_km

logger = logging.getLogger('slt_ai.clustering')

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    _SKL_AVAILABLE = True
except ImportError:
    logger.warning("scikit-learn not installed — install with: pip install scikit-learn")
    _SKL_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

MODEL_SAVE_PATH = pathlib.Path(Config.MODEL_DIR) / 'kmeans_model.pkl'
META_SAVE_PATH  = pathlib.Path(Config.MODEL_DIR) / 'kmeans_meta.pkl'

# Risk thresholds — faults per cluster as % of total
RISK_HIGH_PCT   = 0.25   # top 25% of total = HIGH
RISK_MEDIUM_PCT = 0.15   # 15–25% = MEDIUM, below 15% = LOW

# Cluster display colours (cycled for up to 10 clusters)
CLUSTER_COLORS = [
    '#00FFD1', '#FF2D78', '#1E90FF', '#FFB020', '#9B59F5',
    '#34D399', '#F87171', '#60A5FA', '#FBBF24', '#A78BFA',
]

# District labels used for nearest-centroid matching
DISTRICT_LABELS = Config.SL_DISTRICTS


class KMeansClusterer:
    """
    Fits K-Means on fault GPS coordinates and annotates each cluster
    with region name, fault count, risk level, and display metadata.
    """

    def __init__(self):
        self._model:   Optional[object] = None
        self._scaler:  Optional[object] = None
        self._meta:    dict             = {}
        self._trained: bool             = False
        self._load_saved_model()

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────────────────────

    def cluster(self, gps_df: pd.DataFrame, n_clusters: int = None) -> dict:
        """
        Main clustering entry point.

        1. Cleans GPS DataFrame
        2. Fits K-Means (or reuses saved model)
        3. Labels each cluster with region name + risk level
        4. Returns structured dict for the Flask endpoint

        Args:
            gps_df:     DataFrame with [latitude, longitude, category, ...].
            n_clusters: Number of clusters (default from Config).

        Returns:
            {
              clusters: [{
                clusterId, regionName, faultCount, density,
                riskLevel, riskScore, topCategory,
                centroid: {lat, lng},
                color, techniciansNeeded
              }],
              totalFaults:      int,
              silhouetteScore:  float,
              dataSource:       str  (set by caller)
            }
        """
        n_clusters = n_clusters or Config.KMEANS_N_CLUSTERS

        if not _SKL_AVAILABLE:
            return self._fallback_clusters(gps_df, n_clusters)

        # ── 1. Clean ─────────────────────────────────────────────────────────
        from data.data_cleaner import DataCleaner
        cleaner = DataCleaner()
        clean_df, stats = cleaner.clean_gps_points(gps_df, min_points=n_clusters * 2)

        if clean_df is None:
            logger.warning("Insufficient GPS data — using fallback clusters")
            return self._fallback_clusters(gps_df, n_clusters)

        # ── 2. Extract feature matrix ─────────────────────────────────────────
        from data.feature_engineer import FeatureEngineer
        fe = FeatureEngineer()
        X  = fe.get_cluster_features(clean_df)      # shape (n, 2)

        # ── 3. Fit K-Means ────────────────────────────────────────────────────
        should_refit = (
            not self._trained
            or self._meta.get('n_clusters')    != n_clusters
            or self._meta.get('training_rows') != len(clean_df)
        )
        if should_refit:
            self._fit(X, n_clusters)

        # ── 4. Assign clusters ────────────────────────────────────────────────
        labels     = self._model.predict(X)
        centroids  = self._model.cluster_centers_   # (k, 2) in original lat/lng space

        clean_df   = clean_df.copy()
        clean_df['cluster_id'] = labels

        # ── 5. Build cluster summaries ────────────────────────────────────────
        sil_score = self._compute_silhouette(X, labels)
        clusters  = self._build_cluster_summaries(
            clean_df, centroids, n_clusters, sil_score
        )

        return {
            'clusters':       clusters,
            'totalFaults':    len(clean_df),
            'silhouetteScore': round(float(sil_score), 3) if sil_score else None,
            'nClusters':      n_clusters,
            'fittedAt':       self._meta.get('fitted_at'),
        }

    def retrain(self, gps_df: pd.DataFrame) -> dict:
        """Force refit and save. Called by POST /api/ai/retrain."""
        from data.data_cleaner     import DataCleaner
        from data.feature_engineer import FeatureEngineer

        cleaner = DataCleaner()
        fe      = FeatureEngineer()

        clean_df, _ = cleaner.clean_gps_points(gps_df, min_points=10)
        if clean_df is None:
            return {'error': 'Insufficient GPS data'}

        X = fe.get_cluster_features(clean_df)
        self._fit(X, Config.KMEANS_N_CLUSTERS)
        return {
            'status':        'refitted',
            'trainingPoints': len(clean_df),
            'nClusters':     Config.KMEANS_N_CLUSTERS,
            'fittedAt':      self._meta.get('fitted_at'),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE — FITTING
    # ─────────────────────────────────────────────────────────────────────────

    def _fit(self, X: np.ndarray, n_clusters: int) -> None:
        """Fit K-Means on feature matrix X."""
        logger.info(f"Fitting K-Means: k={n_clusters}, n_points={len(X)}")

        # Use district centroids as warm-start seeds when available
        init = self._get_district_seeds(n_clusters)

        km = KMeans(
            n_clusters=n_clusters,
            init=init,
            n_init=10 if init == 'k-means++' else 1,
            max_iter=300,
            random_state=Config.KMEANS_RANDOM_STATE,
        )
        km.fit(X)

        self._model   = km
        self._trained = True
        self._meta    = {
            'n_clusters':    n_clusters,
            'training_rows': len(X),
            'inertia':       round(float(km.inertia_), 2),
            'fitted_at':     datetime.utcnow().isoformat() + 'Z',
        }
        self._save_model()
        logger.info(f"K-Means fitted — inertia={self._meta['inertia']}")

    def _get_district_seeds(self, n_clusters: int):
        """
        Return district GPS coordinates as K-Means initialisation seeds.
        Falls back to 'k-means++' if fewer districts than clusters.
        """
        if len(DISTRICT_LABELS) < n_clusters:
            return 'k-means++'
        seeds = np.array([
            [d['lat'], d['lng']]
            for d in DISTRICT_LABELS[:n_clusters]
        ])
        return seeds

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE — CLUSTER ANNOTATION
    # ─────────────────────────────────────────────────────────────────────────

    def _build_cluster_summaries(
        self,
        df: pd.DataFrame,
        centroids: np.ndarray,
        n_clusters: int,
        sil_score: float
    ) -> List[dict]:
        """
        Build a rich summary dict for each cluster.

        For each cluster:
        - Count faults and compute density (faults per km²)
        - Find nearest district name as human-readable label
        - Compute risk level based on fault count % of total
        - Recommend number of technicians needed
        - Assign display colour
        """
        total_faults = len(df)
        summaries    = []

        # Per-cluster fault counts
        cluster_counts = df.groupby('cluster_id').size().to_dict()

        # Category breakdown per cluster
        cat_by_cluster = {}
        if 'category' in df.columns:
            for cid in range(n_clusters):
                subset = df[df['cluster_id'] == cid]
                if not subset.empty:
                    top_cat = subset['category'].value_counts().idxmax()
                    cat_by_cluster[cid] = top_cat
                else:
                    cat_by_cluster[cid] = 'UNKNOWN'

        # Sort clusters by fault count descending (rank for display)
        sorted_clusters = sorted(
            range(n_clusters),
            key=lambda c: cluster_counts.get(c, 0),
            reverse=True
        )

        for rank, cid in enumerate(sorted_clusters):
            centroid_lat = float(centroids[cid][0])
            centroid_lng = float(centroids[cid][1])

            fault_count  = cluster_counts.get(cid, 0)
            fault_pct    = fault_count / max(total_faults, 1)

            # Risk level
            if   fault_pct >= RISK_HIGH_PCT:   risk_level = 'HIGH'
            elif fault_pct >= RISK_MEDIUM_PCT: risk_level = 'MEDIUM'
            else:                              risk_level = 'LOW'
            risk_score = round(fault_pct * 100, 1)

            # Nearest district name
            region_name = self._nearest_district_name(centroid_lat, centroid_lng)

            # Approximate cluster area (bounding box in km²)
            subset = df[df['cluster_id'] == cid]
            density = self._compute_density(subset, fault_count)

            # Technicians recommended: 1 per 20 faults, minimum 1
            techs_needed = max(1, math.ceil(fault_count / 20))

            summaries.append({
                'clusterId':          int(cid),
                'rank':               rank + 1,
                'regionName':         region_name,
                'faultCount':         int(fault_count),
                'faultPercent':       round(fault_pct * 100, 1),
                'riskLevel':          risk_level,
                'riskScore':          risk_score,
                'topCategory':        cat_by_cluster.get(cid, 'UNKNOWN'),
                'centroid': {
                    'lat': round(centroid_lat, 4),
                    'lng': round(centroid_lng, 4),
                },
                'density':            round(density, 2),
                'techniciansNeeded':  techs_needed,
                'color':              CLUSTER_COLORS[rank % len(CLUSTER_COLORS)],
            })

        return summaries

    def _nearest_district_name(self, lat: float, lng: float) -> str:
        """Return the name of the nearest Sri Lanka district to a centroid."""
        best_name = 'Unknown'
        best_dist = float('inf')
        for d in DISTRICT_LABELS:
            dist = haversine_km(lat, lng, d['lat'], d['lng'])
            if dist < best_dist:
                best_dist = dist
                best_name = d['name']
        return best_name

    def _compute_density(self, subset: pd.DataFrame, fault_count: int) -> float:
        """
        Estimate fault density (faults per 100 km²) for a cluster.
        Uses the bounding box area of the cluster's GPS points.
        """
        if subset.empty or fault_count == 0:
            return 0.0
        try:
            lat_range = float(subset['latitude'].max()  - subset['latitude'].min())
            lng_range = float(subset['longitude'].max() - subset['longitude'].min())
            # Convert degrees to km (approximate at SL latitude)
            height_km = lat_range * 111.0
            width_km  = lng_range * 111.0 * math.cos(math.radians(7.5))
            area_km2  = max(height_km * width_km, 1.0)
            return round(fault_count / area_km2 * 100, 2)
        except Exception:
            return 0.0

    def _compute_silhouette(self, X: np.ndarray, labels: np.ndarray) -> Optional[float]:
        """Compute silhouette score (cluster quality, -1 to 1, higher=better)."""
        try:
            n_unique = len(np.unique(labels))
            if n_unique < 2 or len(X) < 10:
                return None
            return float(silhouette_score(X, labels, sample_size=min(len(X), 1000)))
        except Exception:
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE — FALLBACK
    # ─────────────────────────────────────────────────────────────────────────

    def _fallback_clusters(self, gps_df: pd.DataFrame, n_clusters: int) -> dict:
        """
        Rule-based geographic fallback when scikit-learn is unavailable
        or GPS data is insufficient.
        Uses predefined Sri Lanka districts as fixed zones.
        """
        logger.info("Using district-based fallback clusters")

        districts = DISTRICT_LABELS[:n_clusters]
        total = max(len(gps_df) if gps_df is not None else 0, 100)
        base  = total // n_clusters

        clusters = []
        for i, d in enumerate(districts):
            count = base + (total % n_clusters if i == 0 else 0)
            pct   = count / total
            risk  = 'HIGH' if pct >= RISK_HIGH_PCT else ('MEDIUM' if pct >= RISK_MEDIUM_PCT else 'LOW')
            clusters.append({
                'clusterId':         i,
                'rank':              i + 1,
                'regionName':        d['name'],
                'faultCount':        count,
                'faultPercent':      round(pct * 100, 1),
                'riskLevel':         risk,
                'riskScore':         round(pct * 100, 1),
                'topCategory':       'BROADBAND',
                'centroid':          {'lat': d['lat'], 'lng': d['lng']},
                'density':           0.0,
                'techniciansNeeded': max(1, count // 20),
                'color':             CLUSTER_COLORS[i % len(CLUSTER_COLORS)],
            })

        return {
            'clusters':       clusters,
            'totalFaults':    total,
            'silhouetteScore': None,
            'nClusters':      n_clusters,
            'note':           'District-based fallback — install scikit-learn for K-Means',
        }

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE — PERSISTENCE
    # ─────────────────────────────────────────────────────────────────────────

    def _save_model(self) -> None:
        try:
            MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(MODEL_SAVE_PATH, 'wb') as f:
                pickle.dump(self._model, f)
            with open(META_SAVE_PATH, 'wb') as f:
                pickle.dump(self._meta, f)
            logger.info(f"K-Means model saved to {MODEL_SAVE_PATH}")
        except Exception as exc:
            logger.warning(f"K-Means save failed: {exc}")

    def _load_saved_model(self) -> None:
        if not MODEL_SAVE_PATH.exists() or not META_SAVE_PATH.exists():
            return
        try:
            with open(MODEL_SAVE_PATH, 'rb') as f:
                self._model = pickle.load(f)
            with open(META_SAVE_PATH, 'rb') as f:
                self._meta = pickle.load(f)
            self._trained = True
            logger.info(
                f"Loaded saved K-Means model "
                f"(k={self._meta.get('n_clusters')}, "
                f"fitted {self._meta.get('fitted_at', 'unknown')})"
            )
        except Exception as exc:
            logger.warning(f"Could not load K-Means model: {exc}")
