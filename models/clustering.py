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
from models.model_registry import ModelVersionRegistry, build_comparison

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

# SRS 5.6.7 requires a metrics comparison, but its named metrics (MAE, RMSE,
# Accuracy) are Prophet-specific and meaningless for K-Means. Substituting
# the model's own native quality metrics: inertia (lower is better — tighter
# clusters) and silhouette score (higher is better — better-separated
# clusters). See conversation record for this being an explicit judgment
# call, not a spec-stated pair.
_METRICS_LOWER_BETTER  = ['inertia']
_METRICS_HIGHER_BETTER = ['silhouetteScore']

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
        self._registry = ModelVersionRegistry(Config.MODEL_DIR, 'clusterer')
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

        # ── 3. Fit (or refresh if data has changed) — internal freshness
        #      auto-refit, unrelated to admin CSV governance, always was
        #      implicit/automatic. Admin-triggered, governed retraining
        #      goes through retrain() + activate_version() instead
        #      (SRS 5.6.7). ────────────────────────────────────────────────
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

    def cluster_by_exchange_area(self, category_df: pd.DataFrame) -> dict:
        """
        Categorical fallback for CSVs with an EXCHANGEAREA column but no GPS —
        the real WFMS export shape (confirmed against a real sample: exchange-
        area codes like DGD/AD/KY/CEN/MHG/KG, no latitude/longitude at all).

        Not a K-Means variant. `get_cluster_features()` fits raw [lat, lng]
        Euclidean distance, and `_build_cluster_summaries()` computes a
        centroid and a bounding-box density from those coordinates — coercing
        a categorical code through that (e.g. hashing it into a fake number)
        would produce a meaningless centroid/"nearest district"/density for a
        value that was never a coordinate. Exchange has no coordinates in
        this system either (Stage C scaffolding only — see Exchange.java's
        own "No coordinates yet" comment), so there is nothing to geocode a
        code into even if one wanted to. Grouping directly by the code IS the
        zone here — each unique EXCHANGEAREA value is already the ground-
        truth partition, no fitting required.

        Mirrors cluster()'s output shape (same keys per cluster entry) so a
        caller that already knows how to render cluster() results doesn't
        need a second rendering path — centroid/density are genuinely
        `None`, not faked with a placeholder, since no coordinate exists.

        Args:
            category_df: DataFrame with an 'exchange_area' column (see
                UploadManager.to_exchange_area_groups()), optionally
                'category'. 'opmc_code' is ignored here — it rides along in
                the DataFrame only as passthrough metadata, not a grouping
                key (see upload_manager.py's module docstring for why it
                isn't resolved against a real Opmc row).

        Returns:
            {clusters, totalFaults, silhouetteScore: None, nClusters, note}
            — one entry in `clusters` per unique exchange-area code, ranked
            by fault count descending, not capped at Config.KMEANS_N_CLUSTERS
            (unlike cluster(), the number of groups is however many distinct
            codes the data actually has, not a fitted k).
        """
        if category_df is None or category_df.empty or 'exchange_area' not in category_df.columns:
            return {
                'clusters':        [],
                'totalFaults':     0,
                'silhouetteScore': None,
                'nClusters':       0,
                'note':            'No exchange-area rows to group.',
            }

        df = category_df.copy()
        df['exchange_area'] = df['exchange_area'].astype(str).str.strip()
        total_faults = len(df)

        counts = df.groupby('exchange_area').size().sort_values(ascending=False)

        cat_by_group = {}
        if 'category' in df.columns:
            for area in counts.index:
                subset = df[df['exchange_area'] == area]
                cat_by_group[area] = (
                    subset['category'].value_counts().idxmax() if not subset.empty else 'UNKNOWN'
                )

        clusters = []
        for rank, (area, fault_count) in enumerate(counts.items()):
            fault_pct = fault_count / max(total_faults, 1)

            if   fault_pct >= RISK_HIGH_PCT:   risk_level = 'HIGH'
            elif fault_pct >= RISK_MEDIUM_PCT: risk_level = 'MEDIUM'
            else:                              risk_level = 'LOW'

            clusters.append({
                'clusterId':         rank,
                'rank':              rank + 1,
                'regionName':        area,  # the exchange-area code itself — no geocoding exists to resolve a human-readable name
                'faultCount':        int(fault_count),
                'faultPercent':      round(fault_pct * 100, 1),
                'riskLevel':         risk_level,
                'riskScore':         round(fault_pct * 100, 1),
                'topCategory':       cat_by_group.get(area, 'UNKNOWN'),
                'centroid':          None,  # no coordinate exists for an exchange-area code
                'density':           None,  # density needs an area in km², which needs a coordinate
                'techniciansNeeded': max(1, math.ceil(fault_count / 20)),
                'color':             CLUSTER_COLORS[rank % len(CLUSTER_COLORS)],
            })

        return {
            'clusters':        clusters,
            'totalFaults':     total_faults,
            'silhouetteScore': None,  # not a fitted model — no cluster-quality metric applies
            'nClusters':       len(clusters),
            'note':            'Categorical grouping by EXCHANGEAREA — source data has no GPS, '
                                'so this is not K-Means geographic clustering.',
        }

    def retrain(self, gps_df: pd.DataFrame) -> dict:
        """
        SRS 5.6.7 CSV training governance — fits a CANDIDATE K-Means model
        and returns it with a quality-metrics comparison (inertia,
        silhouette score — see _METRICS_* comment above) against the
        currently active version. Does NOT touch the model currently
        serving cluster requests; call activate_version() to promote it.
        Called from app.py's background training job (POST /api/ai/train).
        """
        from data.data_cleaner     import DataCleaner
        from data.feature_engineer import FeatureEngineer

        cleaner = DataCleaner()
        fe      = FeatureEngineer()

        # Same formula cluster()'s own live path already uses (Config.KMEANS_N_CLUSTERS * 2,
        # :134 above) rather than an independently hardcoded 10 — retrain() always fits at
        # Config.KMEANS_N_CLUSTERS (below), so its minimum should track the same source of
        # truth the serving path's minimum does, not drift from it if that constant ever
        # changes. No behavior change at today's default (KMEANS_N_CLUSTERS=5 -> 10, same as
        # the previous hardcoded value) — this closes the "hardcoded, can silently diverge"
        # version of the gap, not a threshold-value bug the way forecasting.py's 30-vs-180 was.
        min_points = Config.KMEANS_N_CLUSTERS * 2
        clean_df, clean_stats = cleaner.clean_gps_points(gps_df, min_points=min_points)
        if clean_df is None:
            return {
                'error': (
                    f"Insufficient GPS data for retraining: "
                    f"{clean_stats.get('output_rows', 0)} points < {min_points} required."
                ),
            }

        X = fe.get_cluster_features(clean_df)
        model, meta = self._fit_model(X, Config.KMEANS_N_CLUSTERS)
        labels = model.labels_
        metrics = {
            'inertia':         meta['inertia'],
            'silhouetteScore': self._compute_silhouette(X, labels),
        }

        candidate  = self._registry.save_version(model, metrics, meta, status='candidate')
        comparison = build_comparison(
            self._registry.get_active(), candidate,
            lower_is_better=_METRICS_LOWER_BETTER,
            higher_is_better=_METRICS_HIGHER_BETTER,
        )

        logger.info(
            f"Candidate version {candidate['versionId']} created — "
            f"inertia={metrics['inertia']}, silhouette={metrics['silhouetteScore']} "
            f"(awaiting Activate Model)"
        )
        return {
            'status':         'candidate',
            'versionId':      candidate['versionId'],
            'trainingPoints': len(clean_df),
            'nClusters':      Config.KMEANS_N_CLUSTERS,
            'metrics':        metrics,
            'comparison':     comparison,
        }

    def activate_version(self, version_id: int) -> dict:
        """Explicit admin action — promotes a candidate to the active, serving model."""
        entry = self._registry.activate(version_id)
        self._model   = self._registry.load_object(version_id)
        self._meta    = entry['meta']
        self._trained = True
        logger.info(f"Clusterer: activated version {version_id}")
        return entry

    def rollback(self, version_id: Optional[int] = None) -> dict:
        """Revert the active model. Defaults to the immediately-previous active version."""
        entry = self._registry.rollback(version_id)
        self._model   = self._registry.load_object(entry['versionId'])
        self._meta    = entry['meta']
        self._trained = True
        logger.info(f"Clusterer: rolled back to version {entry['versionId']}")
        return entry

    def list_versions(self) -> list:
        return self._registry.list_versions()

    def assign_zones(self, df: pd.DataFrame) -> np.ndarray:
        """
        Assign each row's (latitude, longitude) to its nearest fitted
        cluster centroid, using the SAME zones cluster()'s summaries
        describe — without fitting a separate model just for this.

        FR-33 (SRS 5.6.8) uses this to cross-reference historical
        material-usage-by-location (Stage 1's
        DataExtractor.get_material_usage_with_location) against the
        existing demand-cluster zones.

        get_cluster_features() confirms K-Means here is fit directly on
        raw [latitude, longitude] — self._scaler is declared but never
        actually used anywhere in this class — so no scaling step is
        needed to match training-time feature space; a plain
        self._model.predict() on raw lat/lng is exactly what cluster()
        itself does for its own training data (see line where `labels =
        self._model.predict(X)` is called).

        Requires cluster() to have already been called at least once this
        instance's lifetime (or a saved model loaded at init) — raises
        rather than silently predicting against an unfit model, which
        would produce meaningless zone ids.

        Args:
            df: DataFrame with [latitude, longitude] columns.

        Returns:
            numpy array of cluster ids, one per row of df, in row order.
        """
        if not self._trained or self._model is None:
            raise RuntimeError(
                "KMeansClusterer has no fitted model yet — call cluster() first."
            )
        if df is None or df.empty:
            return np.array([], dtype=int)

        from data.feature_engineer import FeatureEngineer
        X = FeatureEngineer().get_cluster_features(df)
        return self._model.predict(X)

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE — FITTING
    # ─────────────────────────────────────────────────────────────────────────

    def _fit(self, X: np.ndarray, n_clusters: int) -> None:
        """
        Fit and immediately activate. Used only for the internal auto-refit
        inside cluster() (keeping live DB/synthetic-sourced clusters
        current) — not the admin-triggered CSV governance flow, so no
        candidate gate applies here. Governed retraining goes through
        retrain() + activate_version() instead (SRS 5.6.7).
        """
        model, meta = self._fit_model(X, n_clusters)
        metrics = {
            'inertia':         meta['inertia'],
            'silhouetteScore': self._compute_silhouette(X, model.labels_),
        }
        self._model   = model
        self._trained = True
        self._meta    = meta
        self._registry.save_version(model, metrics, meta, status='active')
        logger.info(f"K-Means fitted — inertia={meta['inertia']}")

    def _fit_model(self, X: np.ndarray, n_clusters: int):
        """
        Fit a fresh K-Means model on X. Pure — does not mutate
        self._model/self._meta, so it's safe to use for candidate
        evaluation without affecting what's currently serving cluster
        requests.
        """
        logger.info(f"Fitting K-Means: k={n_clusters}, n_points={len(X)}")

        # Use district centroids as warm-start seeds when available
        init = self._get_district_seeds(n_clusters)

        # init is either the literal string 'k-means++' or a numpy array of
        # warm-start seed coordinates — comparing an array to a string with
        # == produces an element-wise array, not a bool, so check the type
        # instead of using == directly.
        using_kmeans_pp = isinstance(init, str) and init == 'k-means++'

        km = KMeans(
            n_clusters=n_clusters,
            init=init,
            n_init=10 if using_kmeans_pp else 1,
            max_iter=300,
            random_state=Config.KMEANS_RANDOM_STATE,
        )
        km.fit(X)

        meta = {
            'n_clusters':    n_clusters,
            'training_rows': len(X),
            'inertia':       round(float(km.inertia_), 2),
            'fitted_at':     datetime.utcnow().isoformat() + 'Z',
        }
        return km, meta

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

    def _load_saved_model(self) -> None:
        """Load the currently active version from the registry, if any."""
        active = self._registry.get_active() or self._migrate_legacy_model()
        if active is None:
            return
        try:
            self._model   = self._registry.load_object(active['versionId'])
            self._meta    = active['meta']
            self._trained = True
            logger.info(
                f"Loaded saved K-Means model (version {active['versionId']}, "
                f"k={self._meta.get('n_clusters')}, "
                f"fitted {self._meta.get('fitted_at', 'unknown')})"
            )
        except Exception as exc:
            logger.warning(f"Could not load K-Means model: {exc}")

    def _migrate_legacy_model(self) -> Optional[dict]:
        """
        One-time import of the pre-versioning fixed-path pickle (if present)
        as version 1/active, so switching to the registry doesn't strand a
        model that was already trained and serving before this change.
        """
        if not MODEL_SAVE_PATH.exists() or not META_SAVE_PATH.exists():
            return None
        try:
            with open(MODEL_SAVE_PATH, 'rb') as f:
                model = pickle.load(f)
            with open(META_SAVE_PATH, 'rb') as f:
                meta = pickle.load(f)
            # Silhouette score can't be recomputed here — the training feature
            # matrix wasn't persisted by the pre-versioning code path, only
            # the fitted model. Comparisons against this one migrated version
            # will show silhouetteScore: None; every version trained after
            # this migration computes it normally.
            metrics = {'inertia': meta.get('inertia'), 'silhouetteScore': None}
            entry = self._registry.save_version(model, metrics, meta, status='active')
            logger.info(
                f"Migrated legacy kmeans_model.pkl into version registry "
                f"as version {entry['versionId']}"
            )
            return entry
        except Exception as exc:
            logger.warning(f"Legacy K-Means model migration failed: {exc}")
            return None
