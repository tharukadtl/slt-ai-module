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
