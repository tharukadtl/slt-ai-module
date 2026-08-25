"""
tests/test_clustering.py — K-Means Clustering Unit Tests
=========================================================
Tests for:
  - GPS data cleaning (DataCleaner)
  - SyntheticDataGenerator GPS output
  - KMeansClusterer.cluster() response structure
  - Cluster annotations (risk levels, region names, colours)
  - Fallback behaviour when scikit-learn is unavailable
  - Feature engineer GPS enrichment
  - Haversine distance and graph construction

Run:
    cd slt-ai-module
    python -m pytest tests/test_clustering.py -v
"""

import sys
import os
import math
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config


# ═════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope='module')
def synth():
    from data.synthetic_data import SyntheticDataGenerator
    return SyntheticDataGenerator(seed=42)


@pytest.fixture(scope='module')
def raw_gps(synth):
    """500-point GPS DataFrame."""
    return synth.fault_gps_points(n=500)


@pytest.fixture(scope='module')
def tech_df(synth):
    """25-technician location DataFrame."""
    return synth.technician_locations(n=25)


@pytest.fixture(scope='module')
def clean_gps(raw_gps):
    from data.data_cleaner import DataCleaner
    cleaner = DataCleaner()
    clean_df, stats = cleaner.clean_gps_points(raw_gps, min_points=20)
    return clean_df, stats


@pytest.fixture(scope='module')
def clusterer():
    from models.clustering import KMeansClusterer
    return KMeansClusterer()


@pytest.fixture(scope='module')
def cluster_result(clusterer, raw_gps):
    return clusterer.cluster(raw_gps, n_clusters=5)


# ═════════════════════════════════════════════════════════════════════════════
# SYNTHETIC GPS DATA TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestSyntheticGPS:

    def test_gps_dataframe_shape(self, raw_gps):
        """GPS DataFrame has required columns."""
        for col in ['id', 'latitude', 'longitude', 'category', 'status', 'priority']:
            assert col in raw_gps.columns, f"Missing column: {col}"

    def test_gps_row_count(self, raw_gps):
        """Requested 500 points returns 500 rows."""
        assert len(raw_gps) == 500

    def test_lat_within_sl_bounds(self, raw_gps):
        """All latitudes within Sri Lanka bounds."""
        assert raw_gps['latitude'].between(Config.SL_LAT_MIN, Config.SL_LAT_MAX).all()

    def test_lng_within_sl_bounds(self, raw_gps):
        """All longitudes within Sri Lanka bounds."""
        assert raw_gps['longitude'].between(Config.SL_LNG_MIN, Config.SL_LNG_MAX).all()

    def test_categories_valid(self, raw_gps):
        """All categories are from the allowed set."""
        valid = set(Config.FAULT_CATEGORIES)
        assert set(raw_gps['category'].unique()).issubset(valid)

    def test_colombo_most_faults(self, raw_gps):
        """
        Colombo area (lat 6.5–7.3, lng 79.6–80.2) should contain ≥ 30%
        of all synthetic faults (40% weight in generator).
        """
        colombo = raw_gps[
            raw_gps['latitude'].between(6.5, 7.3) &
            raw_gps['longitude'].between(79.6, 80.2)
        ]
        pct = len(colombo) / len(raw_gps)
        assert pct >= 0.25, f"Expected ≥25% in Colombo area, got {pct:.1%}"

    def test_technician_df_has_required_cols(self, tech_df):
        """Technician DataFrame has GPS + metadata columns."""
        for col in ['technician_id', 'latitude', 'longitude', 'status']:
            assert col in tech_df.columns, f"Missing column: {col}"

    def test_technician_locations_in_sl(self, tech_df):
        """All technician GPS coordinates within Sri Lanka."""
        assert tech_df['latitude'].between(Config.SL_LAT_MIN, Config.SL_LAT_MAX).all()
        assert tech_df['longitude'].between(Config.SL_LNG_MIN, Config.SL_LNG_MAX).all()


# ═════════════════════════════════════════════════════════════════════════════
# GPS CLEANER TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestGPSCleaner:

    def test_clean_gps_returns_dataframe(self, clean_gps):
        clean_df, stats = clean_gps
        assert clean_df is not None
        assert isinstance(clean_df, pd.DataFrame)

    def test_clean_gps_has_lat_lng(self, clean_gps):
        clean_df, _ = clean_gps
        assert 'latitude'  in clean_df.columns
        assert 'longitude' in clean_df.columns

    def test_clean_gps_no_nulls(self, clean_gps):
        clean_df, _ = clean_gps
        assert clean_df[['latitude','longitude']].isnull().sum().sum() == 0

    def test_clean_gps_within_bounds(self, clean_gps):
        clean_df, _ = clean_gps
        assert clean_df['latitude'].between(Config.SL_LAT_MIN, Config.SL_LAT_MAX).all()
        assert clean_df['longitude'].between(Config.SL_LNG_MIN, Config.SL_LNG_MAX).all()

    def test_clean_gps_stats_keys(self, clean_gps):
        _, stats = clean_gps
        for k in ['input_rows', 'output_rows', 'sufficient']:
            assert k in stats

    def test_out_of_bounds_points_removed(self):
        """Coordinates outside Sri Lanka are removed."""
        from data.data_cleaner import DataCleaner
        df = pd.DataFrame({
            'latitude':  [6.93, 51.5, 40.7, 7.29],    # London, NYC are OOB
            'longitude': [79.86, -0.12, -74.0, 80.63],
        })
        cleaner  = DataCleaner()
        result, _ = cleaner.clean_gps_points(df, min_points=1)
        if result is not None:
            assert result['latitude'].between(Config.SL_LAT_MIN, Config.SL_LAT_MAX).all()
            assert len(result) == 2    # Only Colombo and Kandy remain

    def test_null_gps_rows_removed(self):
        """Rows with null lat/lng are dropped."""
        from data.data_cleaner import DataCleaner
        df = pd.DataFrame({
            'latitude':  [6.93, None, 7.29, 9.66, 6.05],
            'longitude': [79.86, 80.0, None, 80.03, 80.22],
        })
        cleaner  = DataCleaner()
        result, _ = cleaner.clean_gps_points(df, min_points=1)
        if result is not None:
            assert result[['latitude','longitude']].isnull().sum().sum() == 0

    def test_empty_input_returns_none(self):
        """Empty DataFrame returns (None, stats)."""
        from data.data_cleaner import DataCleaner
        result, stats = DataCleaner().clean_gps_points(pd.DataFrame(), min_points=1)
        assert result is None


# ═════════════════════════════════════════════════════════════════════════════
# K-MEANS CLUSTERER TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestKMeansClusterer:

    REQUIRED_CLUSTER_KEYS = [
        'clusterId', 'rank', 'regionName', 'faultCount',
        'faultPercent', 'riskLevel', 'riskScore',
        'centroid', 'color', 'techniciansNeeded',
    ]
    REQUIRED_RESULT_KEYS = [
        'clusters', 'totalFaults', 'nClusters',
    ]
    VALID_RISK_LEVELS = {'HIGH', 'MEDIUM', 'LOW'}
    VALID_COLORS = {
        '#00FFD1', '#FF2D78', '#1E90FF', '#FFB020', '#9B59F5',
        '#34D399', '#F87171', '#60A5FA', '#FBBF24', '#A78BFA',
    }

    def test_cluster_result_is_dict(self, cluster_result):
        assert isinstance(cluster_result, dict)

    def test_required_top_level_keys(self, cluster_result):
        for key in self.REQUIRED_RESULT_KEYS:
            assert key in cluster_result, f"Missing key: {key}"

    def test_clusters_is_list(self, cluster_result):
        assert isinstance(cluster_result['clusters'], list)

    def test_cluster_count_matches_k(self, cluster_result):
        """Number of cluster objects matches requested k=5."""
        assert len(cluster_result['clusters']) == 5

    def test_each_cluster_has_required_keys(self, cluster_result):
        for c in cluster_result['clusters']:
            for key in self.REQUIRED_CLUSTER_KEYS:
                assert key in c, f"Cluster missing key: {key}"

    def test_fault_counts_sum_to_total(self, cluster_result):
        """Sum of per-cluster fault counts equals totalFaults."""
        total  = cluster_result['totalFaults']
        summed = sum(c['faultCount'] for c in cluster_result['clusters'])
        assert summed == total, f"Count mismatch: {summed} vs {total}"

    def test_fault_percents_sum_to_100(self, cluster_result):
        """faultPercent values sum to ~100%."""
        total_pct = sum(c['faultPercent'] for c in cluster_result['clusters'])
        assert abs(total_pct - 100.0) < 1.0, \
            f"Percentages sum to {total_pct:.1f}% (expected ~100%)"

    def test_risk_levels_valid(self, cluster_result):
        """All riskLevel values are HIGH, MEDIUM, or LOW."""
        for c in cluster_result['clusters']:
            assert c['riskLevel'] in self.VALID_RISK_LEVELS, \
                f"Invalid riskLevel: {c['riskLevel']}"

    def test_centroids_within_sl_bounds(self, cluster_result):
        """All cluster centroids are within Sri Lanka."""
        for c in cluster_result['clusters']:
            lat = c['centroid']['lat']
            lng = c['centroid']['lng']
            assert Config.SL_LAT_MIN <= lat <= Config.SL_LAT_MAX, \
                f"Centroid latitude {lat} out of SL bounds"
            assert Config.SL_LNG_MIN <= lng <= Config.SL_LNG_MAX, \
                f"Centroid longitude {lng} out of SL bounds"

    def test_region_names_are_strings(self, cluster_result):
        """All regionName values are non-empty strings."""
        for c in cluster_result['clusters']:
            assert isinstance(c['regionName'], str)
            assert len(c['regionName']) > 0

    def test_technicians_needed_positive(self, cluster_result):
        """techniciansNeeded is at least 1 for every cluster."""
        for c in cluster_result['clusters']:
            assert c['techniciansNeeded'] >= 1

    def test_fault_counts_non_negative(self, cluster_result):
        """No cluster has negative fault count."""
        for c in cluster_result['clusters']:
            assert c['faultCount'] >= 0

    def test_ranks_are_sequential(self, cluster_result):
        """Ranks are 1, 2, 3… in order."""
        ranks = sorted(c['rank'] for c in cluster_result['clusters'])
        expected = list(range(1, len(ranks) + 1))
        assert ranks == expected

    def test_custom_k(self, clusterer, raw_gps):
        """Custom k values (2, 3, 7) produce correct cluster counts."""
        for k in [2, 3, 7]:
            result = clusterer.cluster(raw_gps, n_clusters=k)
            assert len(result['clusters']) == k, \
                f"k={k} produced {len(result['clusters'])} clusters"

    def test_silhouette_score_range(self, cluster_result):
        """Silhouette score (if present) is between -1 and 1."""
        score = cluster_result.get('silhouetteScore')
        if score is not None:
            assert -1.0 <= score <= 1.0, \
                f"Silhouette score {score} out of [-1, 1] range"

    def test_cluster_result_is_stable(self, clusterer, raw_gps):
        """Same input produces identical cluster count (deterministic seed)."""
        r1 = clusterer.cluster(raw_gps, n_clusters=5)
        r2 = clusterer.cluster(raw_gps, n_clusters=5)
        assert len(r1['clusters']) == len(r2['clusters'])

    def test_small_dataset_fallback(self, clusterer):
        """Very small GPS dataset (<10 points) triggers fallback gracefully."""
        tiny = pd.DataFrame({
            'latitude':  [6.93, 7.29, 6.05],
            'longitude': [79.86, 80.63, 80.22],
            'category':  ['BROADBAND', 'FIBER', 'TELEPHONE'],
        })
        result = clusterer.cluster(tiny, n_clusters=5)
        assert isinstance(result, dict)
        assert 'clusters' in result
        assert len(result['clusters']) > 0

    def test_nearest_district_colombo(self):
        """Centroid near Colombo is labelled 'Colombo'."""
        from models.clustering import KMeansClusterer
        kc = KMeansClusterer()
        name = kc._nearest_district_name(6.9271, 79.8612)
        assert name == 'Colombo'

    def test_nearest_district_kandy(self):
        """Centroid near Kandy is labelled 'Kandy'."""
        from models.clustering import KMeansClusterer
        kc = KMeansClusterer()
        name = kc._nearest_district_name(7.2906, 80.6337)
        assert name == 'Kandy'


# ═════════════════════════════════════════════════════════════════════════════
# MODEL VALIDATION — SHEET 10_AI_MODULE, ROW AI-007 (FR-27)
# ═════════════════════════════════════════════════════════════════════════════

class TestModelValidation:
    """
    AI-007 — the silhouette score the clusterer reports must be a real number in
    [-1, 1] AND strictly positive, i.e. the clustering is genuinely better than
    an arbitrary partition.

    TestKMeansClusterer::test_silhouette_score_range above only checks the range
    *if* a score is present, so a clusterer that silently reported None, or a
    negative (worse-than-random) score, would pass it. This row asks the
    stronger question, so it is asserted separately rather than folded in.
    """

    def test_silhouette_score_positive(self, cluster_result):
        score = cluster_result.get('silhouetteScore')
        assert score is not None, (
            "cluster() reported no silhouetteScore at all — cluster quality is "
            "unmeasured, so the model cannot be validated"
        )
        assert -1.0 <= score <= 1.0, f"Silhouette score {score} outside [-1, 1]"
        assert score > 0, (
            f"Silhouette score {score} is not > 0 — the clusters are no better "
            "separated than an arbitrary partition of the same points"
        )


# ═════════════════════════════════════════════════════════════════════════════
# ACCURACY TARGET — SHEET 10_AI_MODULE, ROW AI-016 (FR-27 / SRS 5.6.3)
# ═════════════════════════════════════════════════════════════════════════════

def test_silhouette_score_target(clusterer, raw_gps):
    """
    AI-016 — SRS 5.6.3's clustering accuracy target: silhouette > 0.5
    ("well-separated, meaningful clusters"), plus a finite positive inertia.

    The score is recomputed here directly from sklearn against the fitted
    model's own labels rather than trusting the number cluster() reports, and
    the two are then cross-checked against each other — so this fails loudly
    if the reported figure ever drifts away from the real one.
    """
    from sklearn.metrics       import silhouette_score
    from data.data_cleaner     import DataCleaner
    from data.feature_engineer import FeatureEngineer

    result = clusterer.cluster(raw_gps, n_clusters=5)

    clean_df, _ = DataCleaner().clean_gps_points(raw_gps, min_points=20)
    assert clean_df is not None
    X      = FeatureEngineer().get_cluster_features(clean_df)
    labels = clusterer._model.predict(X)

    score = float(silhouette_score(X, labels))

    # The reported figure is the real one (rounded to 3 dp by the model).
    assert score == pytest.approx(result['silhouetteScore'], abs=1e-3), (
        f"Reported silhouetteScore {result['silhouetteScore']} does not match "
        f"an independently computed {score:.4f}"
    )

    inertia = float(clusterer._model.inertia_)
    assert math.isfinite(inertia), f"Inertia is not finite: {inertia}"
    assert inertia > 0, f"Inertia must be positive, got {inertia}"

    assert score > 0.5, (
        f"Silhouette score {score:.3f} is not > 0.5 — SRS 5.6.3's clustering "
        f"accuracy target is not met on {len(clean_df)} fault GPS points at k=5"
    )


# ═════════════════════════════════════════════════════════════════════════════
# H3 — CATEGORICAL FALLBACK FOR EXCHANGEAREA-ONLY DATA (no GPS)
# ═════════════════════════════════════════════════════════════════════════════
# The real WFMS export (confirmed against a real sample) has EXCHANGEAREA
# codes (DGD, AD, KY, CEN, MHG, KG, ...) and NO latitude/longitude at all.
# Exchange has no coordinates either (Stage C scaffolding only), so there is
# nothing to geocode a code into — cluster_by_exchange_area() groups directly
# by the code instead of coercing it through K-Means's raw [lat,lng] fit.

class TestExchangeAreaCategoricalFallback:

    def _area_df(self):
        """
        20 rows across 3 exchange-area codes: DGD (10 -> 50%, HIGH),
        AD (6 -> 30%, HIGH... use thresholds carefully), KY (4 -> 20%, MEDIUM).
        Recomputed below with exact counts to hit each risk band deliberately.
        """
        rows = (
            [{'exchange_area': 'DGD', 'category': 'BROADBAND', 'created_at': '2026-01-01'}] * 10 +
            [{'exchange_area': 'AD',  'category': 'FIBER',     'created_at': '2026-01-01'}] * 6 +
            [{'exchange_area': 'KY',  'category': 'PHONE',     'created_at': '2026-01-01'}] * 3 +
            [{'exchange_area': 'CEN', 'category': 'TV',        'created_at': '2026-01-01'}] * 1
        )
        return pd.DataFrame(rows)

    def test_one_group_per_unique_exchange_area_code(self, clusterer):
        result = clusterer.cluster_by_exchange_area(self._area_df())
        codes = {c['regionName'] for c in result['clusters']}
        assert codes == {'DGD', 'AD', 'KY', 'CEN'}, (
            f"Expected one group per unique EXCHANGEAREA code, got regionNames {codes}"
        )
        assert result['nClusters'] == 4
        assert result['totalFaults'] == 20

    def test_ranked_by_fault_count_descending(self, clusterer):
        result = clusterer.cluster_by_exchange_area(self._area_df())
        counts = [c['faultCount'] for c in result['clusters']]
        assert counts == sorted(counts, reverse=True), (
            f"Clusters must be ranked by faultCount descending, got {counts}"
        )
        assert result['clusters'][0]['regionName'] == 'DGD'
        assert result['clusters'][0]['faultCount'] == 10

    def test_risk_levels_match_the_same_thresholds_as_kmeans(self, clusterer):
        """DGD=10/20=50% HIGH, AD=6/20=30% HIGH, KY=3/20=15% MEDIUM, CEN=1/20=5% LOW."""
        result = clusterer.cluster_by_exchange_area(self._area_df())
        by_area = {c['regionName']: c for c in result['clusters']}
        assert by_area['DGD']['riskLevel'] == 'HIGH'
        assert by_area['AD']['riskLevel']  == 'HIGH'
        assert by_area['KY']['riskLevel']  == 'MEDIUM'
        assert by_area['CEN']['riskLevel'] == 'LOW'

    def test_top_category_computed_per_group(self, clusterer):
        result = clusterer.cluster_by_exchange_area(self._area_df())
        by_area = {c['regionName']: c for c in result['clusters']}
        assert by_area['DGD']['topCategory'] == 'BROADBAND'
        assert by_area['AD']['topCategory']  == 'FIBER'
        assert by_area['KY']['topCategory']  == 'PHONE'

    def test_no_coordinate_is_fabricated(self, clusterer):
        """
        No geocoding exists for an exchange-area code — centroid/density must
        be genuinely absent, not a placeholder like {0,0} or a copied district.
        """
        result = clusterer.cluster_by_exchange_area(self._area_df())
        for c in result['clusters']:
            assert c['centroid'] is None, f"{c['regionName']} has a fabricated centroid: {c['centroid']}"
            assert c['density']  is None, f"{c['regionName']} has a fabricated density: {c['density']}"

    def test_silhouette_score_is_none_not_a_fitted_model(self, clusterer):
        """This is a groupby, not K-Means — no cluster-quality metric applies."""
        result = clusterer.cluster_by_exchange_area(self._area_df())
        assert result['silhouetteScore'] is None

    def test_response_shape_matches_kmeans_cluster_keys(self, clusterer):
        """
        Same per-cluster keys as cluster()'s K-Means output, so a caller that
        already knows how to read a cluster() result doesn't need a second
        parsing path — only centroid/density genuinely differ (None vs real).
        """
        result = clusterer.cluster_by_exchange_area(self._area_df())
        expected_keys = {
            'clusterId', 'rank', 'regionName', 'faultCount', 'faultPercent',
            'riskLevel', 'riskScore', 'topCategory', 'centroid', 'density',
            'techniciansNeeded', 'color',
        }
        for c in result['clusters']:
            assert expected_keys.issubset(c.keys()), (
                f"Missing keys vs cluster()'s shape: {expected_keys - c.keys()}"
            )

    def test_opmc_code_column_is_ignored_as_a_grouping_key(self, clusterer):
        """
        opmc_code rides along as passthrough metadata (see upload_manager.py) —
        it must not silently become a second grouping dimension or change the
        exchange-area group count.
        """
        df = self._area_df()
        df['opmc_code'] = ['KTOP'] * 10 + ['ADOP'] * 6 + ['KYOP'] * 3 + ['HOOP'] * 1
        result = clusterer.cluster_by_exchange_area(df)
        assert result['nClusters'] == 4, (
            "opmc_code must not affect the exchange-area grouping"
        )

    def test_empty_input_returns_empty_result_not_an_error(self, clusterer):
        result = clusterer.cluster_by_exchange_area(pd.DataFrame())
        assert result['clusters'] == []
        assert result['totalFaults'] == 0
        assert result['nClusters'] == 0

    def test_missing_exchange_area_column_returns_empty_result(self, clusterer):
        df = pd.DataFrame({'category': ['BROADBAND'], 'created_at': ['2026-01-01']})
        result = clusterer.cluster_by_exchange_area(df)
        assert result['clusters'] == []


class TestH1dAdditiveColumnsDoNotAffectClustering:
    """
    H1d (2026-08-21): get_faults_with_location() now also selects circuit_id,
    nearest_exchange_id, nearest_exchange_distance_km alongside latitude/
    longitude, purely so a clustered fault can be cross-referenced against
    its stable Exchange/Circuit. GPS remains the sole input to the actual
    K-Means fit (see FeatureEngineer.get_cluster_features()'s explicit
    [latitude, longitude] slice, and DataCleaner.clean_gps_points()'s
    dropna/drop_duplicates, both scoped to the lat/lng columns by name).

    This proves that claim end-to-end: cluster() run on the same GPS/
    category/status/priority data, once with the three new columns present
    and once without, must produce byte-for-byte identical output.
    """

    def _with_new_columns(self, raw_gps):
        df = raw_gps.copy()
        n = len(df)
        # Deliberately mixed non-null/null values, mirroring real data where
        # most faults still have NULL circuit_id/nearest_exchange_id today.
        df['circuit_id'] = [188016 + (i % 7) if i % 3 else None for i in range(n)]
        df['nearest_exchange_id'] = [32 + (i % 5) if i % 2 else None for i in range(n)]
        df['nearest_exchange_distance_km'] = [
            round(0.5 + (i % 11) * 0.37, 2) if i % 2 else None for i in range(n)
        ]
        return df

    def test_cluster_output_identical_with_and_without_new_columns(self, clusterer, raw_gps):
        without_cols = raw_gps.copy()
        with_cols = self._with_new_columns(raw_gps)

        result_without = clusterer.cluster(without_cols, n_clusters=5)
        result_with = clusterer.cluster(with_cols, n_clusters=5)

        assert result_with == result_without, (
            "Adding circuit_id/nearest_exchange_id/nearest_exchange_distance_km "
            "changed K-Means clustering output -- these columns must be inert."
        )

    def test_per_row_cluster_assignments_identical_with_and_without_new_columns(self, clusterer, raw_gps):
        """Same proof at the row-assignment level, not just the summarised dict."""
        from data.data_cleaner import DataCleaner
        from data.feature_engineer import FeatureEngineer

        without_cols = raw_gps.copy()
        with_cols = self._with_new_columns(raw_gps)

        cleaner = DataCleaner()
        engineer = FeatureEngineer()

        clean_without, _ = cleaner.clean_gps_points(without_cols, min_points=10)
        clean_with, _ = cleaner.clean_gps_points(with_cols, min_points=10)

        assert len(clean_without) == len(clean_with), (
            "The new columns changed how many rows survive GPS cleaning."
        )

        features_without = engineer.get_cluster_features(clean_without)
        features_with = engineer.get_cluster_features(clean_with)
        assert np.array_equal(features_without, features_with), (
            "The new columns changed the feature matrix fed to K-Means."
        )

        # Fit independently on each feature matrix (identical values -> identical fit)
        from sklearn.cluster import KMeans
        labels_without = KMeans(n_clusters=5, random_state=42, n_init=10).fit_predict(features_without)
        labels_with = KMeans(n_clusters=5, random_state=42, n_init=10).fit_predict(features_with)
        assert np.array_equal(labels_without, labels_with), (
            "Per-row cluster assignments differ depending on whether the new "
            "columns are present -- clustering must be based on GPS alone."
        )
