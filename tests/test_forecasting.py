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
    """
    220-day raw time-series DataFrame — comfortable headroom above
    Config.FORECAST_MIN_HISTORY_DAYS (180): cleaning (dedup/outlier removal)
    can drop rows, and a fixture sized exactly at the minimum would flip
    forecast() into its insufficientData path on any row loss at all.
    """
    return synth.fault_time_series(days=220)


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
        """220-day request produces 220 rows."""
        assert len(raw_ts) == 220

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


# ═════════════════════════════════════════════════════════════════════════════
# MODEL VALIDATION — SHEET 10_AI_MODULE, ROW AI-003 (FR-26)
# ═════════════════════════════════════════════════════════════════════════════

class TestModelValidation:
    """
    AI-003 — the sheet's stated Prophet accuracy targets, measured on a real
    out-of-sample holdout rather than in-sample: train on the first 150 days of
    a seeded 180-day series, predict the remaining 30, and score the prediction
    against the actual values that were never shown to the model.

    Deliberately does NOT reuse ProphetForecaster.forecast()'s own reported
    `metrics` block — those are the in-sample/cross-validation numbers the model
    computes about itself. The row asks whether the model is accurate, so MAE
    and accuracy (100 − MAPE) are recomputed here from the raw predictions.

    `_fit_model()` is used rather than `forecast()` because it is documented as
    pure (it does not mutate self._model/self._meta), so this test cannot
    disturb the model currently serving the other tests in this file.
    """

    TRAIN_DAYS   = 150
    HOLDOUT_DAYS = 30
    MAE_TARGET      = 5.0
    ACCURACY_TARGET = 85.0

    def test_mae_lt5_accuracy_gte85(self, synth):
        from models.forecasting    import ProphetForecaster, _PROPHET_AVAILABLE
        from data.data_cleaner     import DataCleaner
        from data.feature_engineer import FeatureEngineer

        if not _PROPHET_AVAILABLE:
            pytest.skip("Prophet is not installed — model validation cannot run")

        full     = synth.fault_time_series(days=self.TRAIN_DAYS + self.HOLDOUT_DAYS)
        train_raw = full.iloc[:self.TRAIN_DAYS].copy()
        holdout   = full.iloc[self.TRAIN_DAYS:][['ds', 'y']].copy()
        assert len(holdout) == self.HOLDOUT_DAYS

        clean_df, _ = DataCleaner().clean_time_series(train_raw, min_rows=30)
        assert clean_df is not None, "150 days of seeded data must survive cleaning"

        fe        = FeatureEngineer()
        train_df  = fe.add_prophet_regressors(clean_df)
        model, _  = ProphetForecaster()._fit_model(train_df)

        future    = model.make_future_dataframe(periods=self.HOLDOUT_DAYS, freq='D')
        enriched  = fe.add_regressors_to_future(future, float(train_df['y'].mean()))
        predicted = model.predict(enriched)

        cutoff = train_df['ds'].max()
        pred   = predicted[predicted['ds'] > cutoff][['ds', 'yhat']]
        scored = pred.merge(holdout, on='ds', how='inner')
        assert len(scored) == self.HOLDOUT_DAYS, (
            f"Expected {self.HOLDOUT_DAYS} scoreable holdout days, got {len(scored)} — "
            "the predicted window and the holdout window do not line up"
        )

        abs_err  = (scored['yhat'] - scored['y']).abs()
        mae      = float(abs_err.mean())
        mape     = float((abs_err / scored['y'].clip(lower=1)).mean() * 100)
        accuracy = 100.0 - mape

        failures = []
        if not mae < self.MAE_TARGET:
            failures.append(
                f"MAE {mae:.2f} is not < {self.MAE_TARGET} "
                f"(mean actual daily faults over the holdout: {scored['y'].mean():.1f})"
            )
        if not accuracy >= self.ACCURACY_TARGET:
            failures.append(
                f"Accuracy {accuracy:.1f}% is not >= {self.ACCURACY_TARGET}% "
                f"(MAPE {mape:.1f}%)"
            )
        assert not failures, (
            "Prophet 30-day out-of-sample holdout misses the sheet's targets:\n  "
            + "\n  ".join(failures)
        )


# ═════════════════════════════════════════════════════════════════════════════
# CSV TRAINING GOVERNANCE — SHEET 10_AI_MODULE, ROW AI-023 (FR-30 / SRS 5.6.7)
# ═════════════════════════════════════════════════════════════════════════════

def test_retrain_creates_candidate_not_active(synth, tmp_path):
    """
    AI-023 — retraining must produce a CANDIDATE and leave whatever is already
    ACTIVE serving live traffic untouched; promotion happens only through the
    explicit Activate Model gate (AI-024).

    Runs against a throwaway ModelVersionRegistry rooted in tmp_path so the real
    models/saved/versions/forecaster registry (which the running service loads
    on boot) is neither read nor written by this test.
    """
    from models.forecasting    import ProphetForecaster
    from models.model_registry import ModelVersionRegistry

    forecaster = ProphetForecaster()
    forecaster._registry = ModelVersionRegistry(tmp_path, 'forecaster')

    # Pre-condition from the row: an ACTIVE version already exists. The stored
    # object is never loaded by retrain() (only by activate_version()), so a
    # plain picklable stand-in is enough and avoids fitting a second Prophet.
    previous = forecaster._registry.save_version(
        {'stand-in': 'previously activated forecaster'},
        {'mae': 9.9, 'rmse': 11.1, 'accuracy': 60.0},
        {'training_rows': 150},
        status='active',
    )

    result = forecaster.retrain(synth.fault_time_series(days=180))
    assert 'error' not in result, f"retrain() failed outright: {result}"

    # 1. The new version is a candidate, by its own report and in the registry.
    assert result['status'] == 'candidate'
    by_id = {v['versionId']: v for v in forecaster._registry.list_versions()}
    assert by_id[result['versionId']]['status'] == 'candidate', \
        f"New version {result['versionId']} must be 'candidate', not auto-promoted"

    # 2. The PREVIOUS version is still the one serving traffic.
    assert by_id[previous['versionId']]['status'] == 'active'
    assert forecaster._registry.get_active()['versionId'] == previous['versionId'], \
        "Training silently changed which version is active — the Activate gate is decorative"

    # 3. The comparison object reports new-vs-active deltas.
    comparison = result['comparison']
    assert comparison['previousVersionId'] == previous['versionId']
    assert comparison['delta']['mae']['previous'] == 9.9
    assert comparison['delta']['mae']['candidate'] == result['mae']
    assert isinstance(comparison['delta']['mae']['improved'], bool)


# ═════════════════════════════════════════════════════════════════════════════
# AUTO-REFIT DATE STALENESS — SHEET 10_AI_MODULE, ROWS AI-001/AI-004
# ═════════════════════════════════════════════════════════════════════════════

def test_stale_model_triggers_auto_refit_on_forecast(synth, tmp_path):
    """
    AI-001/AI-004 — forecast()'s internal freshness auto-refit
    (should_retrain) must fire on date staleness alone, not just row-count
    growth. Before this fix, a model trained on a full window that then just
    sat unrefreshed while calendar time passed would never retrain (row
    count never grew), so make_future_dataframe() kept extending from that
    stale training date while _build_response()'s cutoff check compared
    against today — the forecast silently shrank toward empty the longer the
    model went untouched.

    Also verifies the *mechanism* of the fix, not just the symptom: the
    auto-triggered refit must go through save_version(candidate) ->
    activate_version() — the same promotion path retrain()/Activate Model
    uses — rather than a direct save_version(status='active') write, so the
    previous active version is archived by the same activate() call every
    other promotion in this system goes through.

    Runs against a throwaway ModelVersionRegistry rooted in tmp_path, same
    isolation pattern as test_retrain_creates_candidate_not_active, so the
    real models/saved/versions/forecaster registry the running service loads
    on boot is neither read nor written by this test.
    """
    from models.forecasting    import ProphetForecaster
    from models.model_registry import ModelVersionRegistry
    from config                import Config

    forecaster = ProphetForecaster()
    forecaster._registry = ModelVersionRegistry(tmp_path, 'forecaster')
    # __init__ already loaded whatever's active in the REAL registry before
    # the swap above — reset so this instance starts genuinely untrained
    # against the fresh, empty tmp_path registry instead of inheriting
    # possibly-just-refreshed real state.
    forecaster._trained = False
    forecaster._model   = None
    forecaster._meta    = {}

    raw = synth.fault_time_series(days=220)

    # First call — nothing active yet, should_retrain() fires on
    # `not self._trained`, producing and activating version 1.
    result = forecaster.forecast(raw, horizon=7)
    assert len(result['forecast']) == 7
    first_version = forecaster._registry.get_active()['versionId']

    # Backdate the now-active version's last_trained past the staleness
    # window, mirroring what happens over real elapsed time without waiting
    # Config.RETRAIN_INTERVAL_HOURS hours. get_active()['meta'] is the exact
    # same dict object as forecaster._meta (both point at the registry
    # entry's meta), so mutating one is visible through the other.
    stale_ts = (
        datetime.utcnow() - timedelta(hours=Config.RETRAIN_INTERVAL_HOURS + 1)
    ).isoformat() + 'Z'
    forecaster._meta['last_trained'] = stale_ts

    # Same raw data, same row count — isolates staleness as the trigger,
    # since the existing row-count-growth check alone would not fire here.
    result2 = forecaster.forecast(raw, horizon=7)
    second_version = forecaster._registry.get_active()['versionId']
    assert second_version != first_version, \
        "Staleness alone must trigger a refit even with no row-count growth"
    assert len(result2['forecast']) == 7, \
        "Post-refit forecast must be full length again, not shrunk by staleness"

    versions = {v['versionId']: v for v in forecaster._registry.list_versions()}
    assert versions[second_version]['status'] == 'active'
    assert versions[first_version]['status'] == 'archived', \
        "Auto-refit must archive the previous active version via activate(), " \
        "the same as every other promotion path in this system"

    # A genuinely fresh model (last_trained just set by the refit above)
    # must NOT be refit again on the very next request.
    result3 = forecaster.forecast(raw, horizon=7)
    third_version = forecaster._registry.get_active()['versionId']
    assert third_version == second_version, \
        "A genuinely fresh model must not be refit unnecessarily on every request"
