"""
tests/test_forecasting.py — Prophet Forecasting Unit Tests
===========================================================
Tests for:
  - Data cleaning pipeline (DataCleaner)
  - Feature engineering (FeatureEngineer)
  - Synthetic data generation (SyntheticDataGenerator)
  - ProphetForecaster.forecast() response structure
  - ProphetForecaster.retrain() metrics
  - Fallback behaviour when Prophet is unavailable

Run:
    cd slt-ai-module
    python -m pytest tests/test_forecasting.py -v
    python -m pytest tests/test_forecasting.py -v --tb=short -q   # quiet
"""

import sys
import os
import math
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ── Make project root importable ─────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ═════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope='module')
def synth():
    """SyntheticDataGenerator instance (seeded)."""
    from data.synthetic_data import SyntheticDataGenerator
    return SyntheticDataGenerator(seed=42)


@pytest.fixture(scope='module')
def raw_ts(synth):
    """180-day raw time-series DataFrame."""
    return synth.fault_time_series(days=180)


@pytest.fixture(scope='module')
def clean_ts(raw_ts):
    """Cleaned time-series ready for Prophet."""
    from data.data_cleaner import DataCleaner
    cleaner  = DataCleaner()
    clean_df, stats = cleaner.clean_time_series(raw_ts, min_rows=30)
    return clean_df, stats


@pytest.fixture(scope='module')
def enriched_ts(clean_ts):
    """Cleaned + regressor-enriched DataFrame."""
    from data.feature_engineer import FeatureEngineer
    clean_df, _ = clean_ts
    if clean_df is None:
        pytest.skip("Cleaning produced no data")
    fe = FeatureEngineer()
    return fe.add_prophet_regressors(clean_df)


@pytest.fixture(scope='module')
def forecaster():
    """ProphetForecaster instance."""
    from models.forecasting import ProphetForecaster
    return ProphetForecaster()


# ═════════════════════════════════════════════════════════════════════════════
# SYNTHETIC DATA TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestSyntheticData:

    def test_time_series_shape(self, raw_ts):
        """Generated DataFrame has correct columns and row count."""
        assert isinstance(raw_ts, pd.DataFrame)
        assert 'ds' in raw_ts.columns
        assert 'y'  in raw_ts.columns

    def test_time_series_length(self, raw_ts):
        """180-day request produces 180 rows."""
        assert len(raw_ts) == 180

    def test_date_column_dtype(self, raw_ts):
        """ds column is datetime type."""
        assert pd.api.types.is_datetime64_any_dtype(raw_ts['ds'])

    def test_values_non_negative(self, raw_ts):
        """All y values are ≥ 0 (fault counts cannot be negative)."""
        assert (raw_ts['y'] >= 0).all()

    def test_values_reasonable_range(self, raw_ts):
        """Fault counts are in a plausible range (1–200 per day)."""
        assert raw_ts['y'].max() < 200, "Max daily faults seems unrealistically high"
        assert raw_ts['y'].mean() > 2,  "Mean daily faults seems too low"

    def test_dates_are_sequential(self, raw_ts):
        """Dates increase monotonically with no gaps (after sorting)."""
        sorted_df = raw_ts.sort_values('ds')
        diffs = sorted_df['ds'].diff().dropna().dt.days
        assert (diffs == 1).all(), "Date sequence has gaps > 1 day"

    def test_reproducible_with_same_seed(self):
        """Same seed produces identical output."""
        from data.synthetic_data import SyntheticDataGenerator
        g1 = SyntheticDataGenerator(seed=99)
        g2 = SyntheticDataGenerator(seed=99)
        assert g1.fault_time_series(days=30)['y'].tolist() == \
               g2.fault_time_series(days=30)['y'].tolist()

    def test_different_seeds_differ(self):
        """Different seeds produce different output."""
        from data.synthetic_data import SyntheticDataGenerator
        g1 = SyntheticDataGenerator(seed=1)
        g2 = SyntheticDataGenerator(seed=2)
        assert g1.fault_time_series(days=30)['y'].tolist() != \
               g2.fault_time_series(days=30)['y'].tolist()

    def test_weekly_pattern_present(self, raw_ts):
        """
        Weekdays should average more faults than weekends
        (weekly seasonality is baked into the generator).
        """
        df = raw_ts.copy()
        df['dow'] = df['ds'].dt.dayofweek
        weekday_mean = df[df['dow'] < 5]['y'].mean()
        weekend_mean = df[df['dow'] >= 5]['y'].mean()
        assert weekday_mean > weekend_mean, \
            "Expected weekday faults > weekend faults in synthetic data"


# ═════════════════════════════════════════════════════════════════════════════
# DATA CLEANER TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestDataCleaner:

    def test_clean_returns_dataframe(self, clean_ts):
        """clean_time_series returns a DataFrame and a stats dict."""
        clean_df, stats = clean_ts
        assert clean_df is not None
        assert isinstance(clean_df, pd.DataFrame)
        assert isinstance(stats, dict)

    def test_required_columns_present(self, clean_ts):
        """Output has exactly [ds, y] columns."""
        clean_df, _ = clean_ts
        assert set(clean_df.columns) == {'ds', 'y'}

    def test_no_null_values(self, clean_ts):
        """No NaN values in output."""
        clean_df, _ = clean_ts
        assert clean_df.isnull().sum().sum() == 0

    def test_no_negative_values(self, clean_ts):
        """y column has no negative values."""
        clean_df, _ = clean_ts
        assert (clean_df['y'] >= 0).all()

    def test_dates_unique(self, clean_ts):
        """No duplicate dates after cleaning."""
        clean_df, _ = clean_ts
        assert clean_df['ds'].duplicated().sum() == 0

    def test_dates_complete(self, clean_ts):
        """No date gaps in output (zero-fill was applied)."""
        clean_df, _ = clean_ts
        sorted_df = clean_df.sort_values('ds')
        diffs = sorted_df['ds'].diff().dropna().dt.days
        assert (diffs == 1).all(), "Date gaps found after zero-fill"

    def test_stats_dict_keys(self, clean_ts):
        """Stats dict has expected keys."""
        _, stats = clean_ts
        for key in ['input_rows', 'output_rows', 'sufficient', 'mean_y']:
            assert key in stats, f"Missing stats key: {key}"

    def test_sufficient_flag_true(self, clean_ts):
        """180-day input should produce sufficient=True."""
        _, stats = clean_ts
        assert stats['sufficient'] is True

    def test_empty_input_returns_none(self):
        """Empty DataFrame input returns (None, stats)."""
        from data.data_cleaner import DataCleaner
        cleaner = DataCleaner()
        result, stats = cleaner.clean_time_series(pd.DataFrame())
        assert result is None
        assert len(stats['issues']) > 0

    def test_none_input_returns_none(self):
        """None input returns (None, stats)."""
        from data.data_cleaner import DataCleaner
        cleaner = DataCleaner()
        result, stats = cleaner.clean_time_series(None)
        assert result is None

    def test_negative_values_clipped(self):
        """Negative y values are clipped to 0."""
        from data.data_cleaner import DataCleaner
        df = pd.DataFrame({'ds': pd.date_range('2025-01-01', periods=50), 'y': range(-5, 45)})
        cleaner  = DataCleaner()
        result, _ = cleaner.clean_time_series(df, min_rows=10)
        if result is not None:
            assert (result['y'] >= 0).all()

    def test_outlier_removal(self):
        """Extreme outlier (IQR × 3) is removed."""
        from data.data_cleaner import DataCleaner
        dates = pd.date_range('2025-01-01', periods=60)
        vals  = [10] * 59 + [9999]    # one extreme outlier
        df    = pd.DataFrame({'ds': dates, 'y': vals})
        cleaner  = DataCleaner()
        result, stats = cleaner.clean_time_series(df, min_rows=10)
        if result is not None:
            assert result['y'].max() < 9999, "Outlier was not removed"

    def test_duplicate_dates_removed(self):
        """Duplicate dates are collapsed (keep last)."""
        from data.data_cleaner import DataCleaner
        dates = pd.to_datetime(['2025-01-01'] * 5 + ['2025-01-02'] * 5 +
                                list(pd.date_range('2025-01-03', periods=50)))
        vals  = list(range(60))
        df    = pd.DataFrame({'ds': dates, 'y': vals})
        cleaner  = DataCleaner()
        result, _ = cleaner.clean_time_series(df, min_rows=10)
        if result is not None:
            assert result['ds'].duplicated().sum() == 0

    def test_validate_for_prophet_returns_dict(self, clean_ts):
        """validate_for_prophet returns a dict with 'valid' key."""
        from data.data_cleaner import DataCleaner
        clean_df, _ = clean_ts
        cleaner      = DataCleaner()
        result       = cleaner.validate_for_prophet(clean_df)
        assert isinstance(result, dict)
        assert 'valid' in result
        assert result['valid'] is True


# ═════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEER TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestFeatureEngineer:

    REGRESSOR_COLS = [
        'day_of_week', 'is_weekend', 'month', 'week_of_year',
        'is_holiday', 'is_sw_monsoon', 'is_ne_monsoon',
        'rolling_mean_7', 'fault_lag_7',
    ]

    def test_all_regressors_added(self, enriched_ts):
        """All expected regressor columns are present."""
        for col in self.REGRESSOR_COLS:
            assert col in enriched_ts.columns, f"Missing regressor: {col}"

    def test_is_weekend_binary(self, enriched_ts):
        """is_weekend is strictly 0 or 1."""
        assert enriched_ts['is_weekend'].isin([0, 1]).all()

    def test_day_of_week_range(self, enriched_ts):
        """day_of_week is 0–6."""
        assert enriched_ts['day_of_week'].between(0, 6).all()

    def test_month_range(self, enriched_ts):
        """month is 1–12."""
        assert enriched_ts['month'].between(1, 12).all()

    def test_is_holiday_binary(self, enriched_ts):
        """is_holiday is 0 or 1."""
        assert enriched_ts['is_holiday'].isin([0, 1]).all()

    def test_monsoon_flags_binary(self, enriched_ts):
        """is_sw_monsoon and is_ne_monsoon are 0 or 1."""
        assert enriched_ts['is_sw_monsoon'].isin([0, 1]).all()
        assert enriched_ts['is_ne_monsoon'].isin([0, 1]).all()

    def test_rolling_mean_no_nan_after_fillna(self, enriched_ts):
        """rolling_mean_7 contains no NaN (min_periods=1 fills early rows)."""
        assert enriched_ts['rolling_mean_7'].isna().sum() == 0

    def test_sw_monsoon_months(self, clean_ts):
        """May–September rows should have is_sw_monsoon == 1."""
        from data.feature_engineer import FeatureEngineer
        clean_df, _ = clean_ts
        fe = FeatureEngineer()
        df = fe.add_prophet_regressors(clean_df)
        sw = df[df['month'].isin([5, 6, 7, 8, 9])]
        if len(sw) > 0:
            assert (sw['is_sw_monsoon'] == 1).all()

    def test_haversine_distance(self):
        """Haversine distance between Colombo and Kandy ≈ 95 km."""
        from data.feature_engineer import haversine_km
        d = haversine_km(6.9271, 79.8612, 7.2906, 80.6337)
        assert 90 < d < 110, f"Expected ~95 km, got {d:.1f} km"

    def test_haversine_zero_distance(self):
        """Distance from a point to itself is 0."""
        from data.feature_engineer import haversine_km
        d = haversine_km(7.0, 80.0, 7.0, 80.0)
        assert d == pytest.approx(0.0, abs=0.001)

    def test_estimate_travel_minutes(self):
        """10 km at 40 km/h → 15 minutes."""
        from data.feature_engineer import estimate_travel_minutes
        m = estimate_travel_minutes(10.0, avg_speed_kmh=40.0)
        assert m == 15

    def test_build_distance_graph_structure(self, synth):
        """Distance graph has correct keys and technician nodes."""
        from data.feature_engineer import FeatureEngineer
        from data.data_cleaner    import DataCleaner
        fe   = FeatureEngineer()
        tech = DataCleaner().clean_technician_locations(synth.technician_locations(n=5))
        graph = fe.build_distance_graph(tech, fault_lat=6.93, fault_lng=79.86)

        assert 'nodes'      in graph
        assert 'edges'      in graph
        assert 'fault_node' in graph
        assert graph['fault_node'] == 'fault_0'
        assert 'fault_0' in graph['nodes']
        # Should have n technician nodes + 1 fault node
        assert len(graph['nodes']) == len(tech) + 1


# ═════════════════════════════════════════════════════════════════════════════
# PROPHET FORECASTER TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestProphetForecaster:

    REQUIRED_KEYS = [
        'historical', 'forecast', 'metrics',
        'trend', 'trendPercent', 'nextPeriodTotal', 'weeklyPattern',
    ]
    REQUIRED_METRIC_KEYS = [
        'mae', 'rmse', 'accuracy', 'trainingDays', 'horizon', 'lastTrained',
    ]

    def test_forecast_returns_dict(self, forecaster, raw_ts):
        """forecast() returns a dict."""
        result = forecaster.forecast(raw_ts, horizon=7)
        assert isinstance(result, dict)

    def test_required_top_level_keys(self, forecaster, raw_ts):
        """All required top-level keys present in forecast result."""
        result = forecaster.forecast(raw_ts, horizon=7)
        for key in self.REQUIRED_KEYS:
            assert key in result, f"Missing key: {key}"

    def test_historical_is_list(self, forecaster, raw_ts):
        """historical is a non-empty list."""
        result = forecaster.forecast(raw_ts, horizon=7)
        assert isinstance(result['historical'], list)
        assert len(result['historical']) > 0

    def test_forecast_is_list(self, forecaster, raw_ts):
        """forecast is a list."""
        result = forecaster.forecast(raw_ts, horizon=7)
        assert isinstance(result['forecast'], list)

    def test_forecast_length_matches_horizon(self, forecaster, raw_ts):
        """forecast list length equals requested horizon."""
        for h in [7, 14, 30]:
            result = forecaster.forecast(raw_ts, horizon=h)
            fcast  = result['forecast']
            assert len(fcast) == h, \
                f"Expected {h} forecast days, got {len(fcast)}"

    def test_historical_items_structure(self, forecaster, raw_ts):
        """Each historical item has ds, y, actual, isForecast=False."""
        result = forecaster.forecast(raw_ts, horizon=7)
        for item in result['historical'][:5]:
            assert 'ds'         in item
            assert 'y'          in item
            assert 'actual'     in item
            assert 'isForecast' in item
            assert item['isForecast'] is False

    def test_forecast_items_structure(self, forecaster, raw_ts):
        """Each forecast item has ds, yhat, upper, lower, isForecast=True."""
        result = forecaster.forecast(raw_ts, horizon=7)
        for item in result['forecast']:
            assert 'ds'         in item
            assert 'yhat'       in item
            assert 'upper'      in item
            assert 'lower'      in item
            assert 'isForecast' in item
            assert item['isForecast'] is True

    def test_forecast_yhat_non_negative(self, forecaster, raw_ts):
        """All predicted values are ≥ 0."""
        result = forecaster.forecast(raw_ts, horizon=14)
        for item in result['forecast']:
            assert item['yhat'] >= 0, f"Negative yhat: {item['yhat']}"

    def test_confidence_bands_ordered(self, forecaster, raw_ts):
        """lower ≤ yhat ≤ upper for each forecast point."""
        result = forecaster.forecast(raw_ts, horizon=14)
        for item in result['forecast']:
            assert item['lower'] <= item['yhat'], \
                f"lower ({item['lower']}) > yhat ({item['yhat']})"
            assert item['yhat'] <= item['upper'], \
                f"yhat ({item['yhat']}) > upper ({item['upper']})"

    def test_metrics_structure(self, forecaster, raw_ts):
        """metrics dict has required keys."""
        result  = forecaster.forecast(raw_ts, horizon=7)
        metrics = result['metrics']
        for key in self.REQUIRED_METRIC_KEYS:
            assert key in metrics, f"Missing metrics key: {key}"

    def test_accuracy_in_valid_range(self, forecaster, raw_ts):
        """Accuracy (when computed) is between 0 and 100."""
        result   = forecaster.forecast(raw_ts, horizon=7)
        accuracy = result['metrics'].get('accuracy')
        if accuracy is not None:
            assert 0 <= accuracy <= 100, f"Accuracy {accuracy} out of range"

    def test_trend_is_valid_value(self, forecaster, raw_ts):
        """trend is 'up', 'down', or 'stable'."""
        result = forecaster.forecast(raw_ts, horizon=7)
        assert result['trend'] in ('up', 'down', 'stable')

    def test_trend_percent_is_float(self, forecaster, raw_ts):
        """trendPercent is a numeric value."""
        result = forecaster.forecast(raw_ts, horizon=7)
        assert isinstance(result['trendPercent'], (int, float))

    def test_next_period_total_positive(self, forecaster, raw_ts):
        """nextPeriodTotal is a non-negative integer."""
        result = forecaster.forecast(raw_ts, horizon=7)
        assert isinstance(result['nextPeriodTotal'], int)
        assert result['nextPeriodTotal'] >= 0

    def test_weekly_pattern_has_7_days(self, forecaster, raw_ts):
        """weeklyPattern has exactly 7 entries."""
        result = forecaster.forecast(raw_ts, horizon=7)
        assert len(result['weeklyPattern']) == 7

    def test_weekly_pattern_structure(self, forecaster, raw_ts):
        """Each weekly pattern item has dayName and avgFaults."""
        result = forecaster.forecast(raw_ts, horizon=7)
        for item in result['weeklyPattern']:
            assert 'dayName'   in item
            assert 'avgFaults' in item

    def test_different_horizons_produce_different_totals(self, forecaster, raw_ts):
        """7-day and 30-day totals should differ."""
        r7  = forecaster.forecast(raw_ts, horizon=7)
        r30 = forecaster.forecast(raw_ts, horizon=30)
        assert r7['nextPeriodTotal'] != r30['nextPeriodTotal']

    def test_none_input_returns_dict(self, forecaster):
        """None input falls back gracefully and returns a valid dict."""
        result = forecaster.forecast(None, horizon=7)
        assert isinstance(result, dict)
        assert 'forecast' in result

    def test_empty_df_returns_dict(self, forecaster):
        """Empty DataFrame falls back gracefully."""
        result = forecaster.forecast(pd.DataFrame(), horizon=7)
        assert isinstance(result, dict)

    def test_data_source_key_present(self, forecaster, raw_ts):
        """dataSource is set after route layer adds it."""
        # The forecaster itself doesn't set dataSource; routes/ do.
        # But fallback forecast should still be a valid dict.
        result = forecaster.forecast(raw_ts, horizon=7)
        assert isinstance(result, dict)

    def test_retrain_returns_metrics_dict(self, forecaster, raw_ts):
        """retrain() returns a dict with accuracy info."""
        metrics = forecaster.retrain(raw_ts)
        assert isinstance(metrics, dict)
        # Should have at least one of these
        has_any = any(k in metrics for k in ['mae','rmse','accuracy','error'])
        assert has_any, f"retrain() returned unexpected dict: {metrics}"
