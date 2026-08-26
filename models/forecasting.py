"""
models/forecasting.py — Prophet Time-Series Forecasting Model
==============================================================
Trains a Facebook Prophet model on historical daily fault counts
and produces fault volume forecasts with confidence intervals.

SRS requirement:
  - ≥85% accuracy measured by MAE and RMSE
  - 30-day default horizon (configurable 7/14/30/60/90)
  - Trained on 12–24 months of historical fault data
  - Regressors: holidays, weekends, monsoon season, rolling averages

Pipeline:
  1. DataExtractor / SyntheticDataGenerator supplies raw [ds, y] df
  2. DataCleaner validates and zero-fills gaps
  3. FeatureEngineer adds Sri Lanka-specific regressors
  4. ProphetForecaster trains, forecasts, computes metrics, serialises

Usage:
    from models.forecasting import ProphetForecaster
    pf = ProphetForecaster()
    result = pf.forecast(raw_df, horizon=30)
    # result keys: historical, forecast, metrics, trend, nextPeriodTotal
"""

import os
import logging
import math
import pickle
import pathlib
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from config import Config
from models.model_registry import ModelVersionRegistry, build_comparison

logger = logging.getLogger('slt_ai.forecasting')

# ── Prophet is an optional heavy dependency ────────────────────────────────────
try:
    from prophet import Prophet
    from prophet.diagnostics import cross_validation, performance_metrics
    _PROPHET_AVAILABLE = True
except ImportError:
    logger.warning("Prophet not installed — install with: pip install prophet")
    _PROPHET_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

MODEL_SAVE_PATH = pathlib.Path(Config.MODEL_DIR) / 'prophet_model.pkl'
META_SAVE_PATH  = pathlib.Path(Config.MODEL_DIR) / 'prophet_meta.pkl'

# SRS 5.6.7 comparison direction — lower is better for error metrics,
# higher is better for accuracy.
_METRICS_LOWER_BETTER  = ['mae', 'rmse']
_METRICS_HIGHER_BETTER = ['accuracy']

# Prophet hyperparameters (tuned for SLT daily fault data)
PROPHET_PARAMS = {
    'changepoint_prior_scale':  0.15,   # flexibility of trend changepoints
    'seasonality_prior_scale':  10.0,   # flexibility of seasonality components
    'holidays_prior_scale':     10.0,   # weight on holiday effects
    'seasonality_mode':         'multiplicative',  # better for count data with growth
    'yearly_seasonality':       True,
    'weekly_seasonality':       True,
    'daily_seasonality':        False,
    'interval_width':           0.80,   # 80% confidence interval
}

# Additional regressors added by FeatureEngineer
EXTRA_REGRESSORS = [
    'is_holiday',
    'is_weekend',
    'is_sw_monsoon',
    'is_ne_monsoon',
    'rolling_mean_7',
    'fault_lag_7',
]

# Cluster colours for frontend display
TREND_COLORS = {'up': '#EF4444', 'down': '#10B981', 'stable': '#6B7280'}


class ProphetForecaster:
    """
    Wraps Facebook Prophet with SLT-specific preprocessing,
    regressor injection, and model persistence.
    """

    def __init__(self):
        self._model:    Optional[object] = None  # fitted Prophet instance
        self._meta:     dict             = {}    # training metadata + metrics
        self._trained:  bool             = False
        self._registry  = ModelVersionRegistry(Config.MODEL_DIR, 'forecaster')
        self._load_saved_model()

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────────────────────

    def forecast(self, raw_df: pd.DataFrame, horizon: int = None) -> dict:
        """
        Main forecast entry point.

        1. Cleans and validates raw_df
        2. Trains Prophet if not already trained or data changed
        3. Generates forecast for `horizon` days
        4. Returns structured dict for the Flask endpoint

        Args:
            raw_df:  DataFrame with columns [ds, y] (raw daily fault counts).
            horizon: Days to forecast (default from Config).

        Returns:
            {
              historical:       [{ds, y, actual}],        # training data
              forecast:         [{ds, yhat, upper, lower, isForecast}],
              metrics:          {mae, rmse, mape, accuracy, trainingDays,
                                 horizon, lastTrained},
              trend:            "up" | "down" | "stable",
              trendPercent:     float,
              nextPeriodTotal:  int,
              weeklyPattern:    [{dayName, avgFaults}]
            }
        """
        if not _PROPHET_AVAILABLE:
            return self._fallback_forecast(raw_df, horizon or Config.FORECAST_HORIZON_DAYS)

        horizon = horizon or Config.FORECAST_HORIZON_DAYS

        # ── 1. Clean ────────────────────────────────────────────────────────
        from data.data_cleaner     import DataCleaner
        from data.feature_engineer import FeatureEngineer

        cleaner = DataCleaner()
        fe      = FeatureEngineer()

        clean_df, clean_stats = cleaner.clean_time_series(
            raw_df,
            fill_zeros=True,
            remove_outliers=True,
            min_rows=Config.FORECAST_MIN_HISTORY_DAYS,
        )

        if clean_df is None:
            logger.warning(
                f"Insufficient historical data for forecast: "
                f"{clean_stats.get('output_rows', 0)} rows < {Config.FORECAST_MIN_HISTORY_DAYS} required"
            )
            return self._insufficient_data_response(clean_stats, horizon)

        validation = cleaner.validate_for_prophet(clean_df)
        for w in validation.get('warnings', []):
            logger.warning(f"Prophet validation: {w}")

        # ── 2. Add regressors ────────────────────────────────────────────────
        train_df = fe.add_prophet_regressors(clean_df)

        # ── 3. Train (or refresh, on row growth OR staleness) — internal
        #      freshness auto-refit, unrelated to admin CSV governance,
        #      always was implicit/automatic. Admin-triggered, governed
        #      retraining goes through retrain() + activate_version()
        #      instead (SRS 5.6.7). Two independent triggers, either one
        #      refits: (a) enough new rows have accumulated, or (b) the
        #      active model is simply old — AI-001/AI-004: row-count growth
        #      alone never fires for a model that was trained on a full
        #      window and then just sits there while calendar time passes,
        #      so make_future_dataframe() keeps extending from that stale
        #      training date while _build_response()'s cutoff check compares
        #      against today, silently shrinking the forecast by about a day
        #      for every day the model goes unrefreshed. ────────────────────
        should_retrain = (
            not self._trained
            or len(train_df) > self._meta.get('training_rows', 0) + 30
            or self._is_stale()
        )
        if should_retrain:
            self._train(train_df)

        # ── 4. Forecast ──────────────────────────────────────────────────────
        historical_mean = float(train_df['y'].mean())
        future_df = self._model.make_future_dataframe(periods=horizon, freq='D')
        future_enriched = fe.add_regressors_to_future(future_df, historical_mean)
        raw_forecast = self._model.predict(future_enriched)

        # ── 5. Build response ────────────────────────────────────────────────
        return self._build_response(train_df, raw_forecast, horizon)

    def retrain(self, raw_df: pd.DataFrame) -> dict:
        """
        SRS 5.6.7 CSV training governance — fits a CANDIDATE model and returns
        it with a metrics comparison against the currently active version.
        Does NOT touch the model currently serving forecasts; the candidate
        only goes live once activate_version() is explicitly called.
        Called from app.py's background training job (POST /api/ai/train).
        """
        from data.data_cleaner     import DataCleaner
        from data.feature_engineer import FeatureEngineer

        cleaner = DataCleaner()
        fe      = FeatureEngineer()

        clean_df, clean_stats = cleaner.clean_time_series(
            raw_df, min_rows=Config.FORECAST_MIN_HISTORY_DAYS,
        )
        if clean_df is None:
            return {
                'error': (
                    f"Insufficient data for retraining: "
                    f"{clean_stats.get('output_rows', 0)} days < "
                    f"{Config.FORECAST_MIN_HISTORY_DAYS} required."
                ),
            }

        train_df = fe.add_prophet_regressors(clean_df)
        model, meta = self._fit_model(train_df)
        metrics = self._compute_cv_metrics(train_df, model=model, meta=meta)
        meta['cv_metrics'] = metrics

        candidate  = self._registry.save_version(model, metrics, meta, status='candidate')
        comparison = build_comparison(
            self._registry.get_active(), candidate,
            lower_is_better=_METRICS_LOWER_BETTER,
            higher_is_better=_METRICS_HIGHER_BETTER,
        )

        logger.info(
            f"Candidate version {candidate['versionId']} created — "
            f"MAE={metrics.get('mae')}, RMSE={metrics.get('rmse')} (awaiting Activate Model)"
        )
        return {
            **metrics,
            'versionId':  candidate['versionId'],
            'status':     'candidate',
            'comparison': comparison,
        }

    def activate_version(self, version_id: int) -> dict:
        """Explicit admin action — promotes a candidate to the active, serving model."""
        entry = self._registry.activate(version_id)
        self._model   = self._registry.load_object(version_id)
        self._meta    = entry['meta']
        self._trained = True
        logger.info(f"Forecaster: activated version {version_id}")
        return entry

    def rollback(self, version_id: Optional[int] = None) -> dict:
        """Revert the active model. Defaults to the immediately-previous active version."""
        entry = self._registry.rollback(version_id)
        self._model   = self._registry.load_object(entry['versionId'])
        self._meta    = entry['meta']
        self._trained = True
        logger.info(f"Forecaster: rolled back to version {entry['versionId']}")
        return entry

    def list_versions(self) -> list:
        return self._registry.list_versions()

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE — TRAINING
    # ─────────────────────────────────────────────────────────────────────────

    def _train(self, train_df: pd.DataFrame) -> None:
        """
        Fit, save as a CANDIDATE, and immediately activate it. Used only for
        the internal freshness auto-refit inside forecast() (keeping live
        DB/synthetic-sourced forecasts current) and initial bootstrap
        training — still not the admin-triggered CSV governance flow (no
        human review is ever required here, this path always was implicit/
        automatic), but it now goes through the same candidate -> activate()
        mechanics retrain()/activate_version() use, rather than a
        save_version(status='active') shortcut, so every promotion —
        automatic or admin-triggered — is auditable through one code path
        and archives the previous active version the same way.
        """
        logger.info(f"Training Prophet on {len(train_df)} days of data…")

        m, meta = self._fit_model(train_df)
        metrics = self._compute_in_sample_metrics(train_df, model=m, meta=meta)
        meta['in_sample_metrics'] = metrics

        candidate  = self._registry.save_version(m, metrics, meta, status='candidate')
        comparison = build_comparison(
            self._registry.get_active(), candidate,
            lower_is_better=_METRICS_LOWER_BETTER,
            higher_is_better=_METRICS_HIGHER_BETTER,
        )
        prev_mae = ((comparison.get('delta') or {}).get('mae') or {}).get('previous')
        logger.info(
            f"Auto-refit: candidate version {candidate['versionId']} created "
            f"(MAE={metrics.get('mae')}, previous active MAE={prev_mae}) — "
            "auto-activating (internal freshness path, no human review required)"
        )
        self.activate_version(candidate['versionId'])

    def _is_stale(self) -> bool:
        """
        AI-001/AI-004 — the other half of should_retrain's trigger, alongside
        row-count growth. True once the active model's own last_trained
        timestamp is more than Config.RETRAIN_INTERVAL_HOURS old.

        last_trained is written as datetime.utcnow().isoformat() + 'Z' (a
        naive UTC timestamp, not a real timezone-aware ISO string), so it's
        parsed back the same way rather than through a timezone-aware path.
        """
        last_trained = self._meta.get('last_trained')
        if not last_trained:
            return False
        trained_at = datetime.fromisoformat(last_trained.rstrip('Z'))
        age_hours = (datetime.utcnow() - trained_at).total_seconds() / 3600
        return age_hours > Config.RETRAIN_INTERVAL_HOURS

    def _fit_model(self, train_df: pd.DataFrame):
        """
        Fit a fresh Prophet model on train_df. Pure — does not mutate
        self._model/self._meta, so it's safe to use for candidate evaluation
        without affecting what's currently serving forecasts.
        """
        # AI-003 (QA_Compliance_Consolidated_Report.md) — an 8-term yearly Fourier
        # series is unidentifiable on under a year of training data and
        # extrapolates away rather than fitting a real pattern. Guard both the
        # built-in yearly_seasonality and the custom yearly_slt override behind
        # the same threshold, rather than fitting a component the data can't
        # support.
        has_full_year = len(train_df) >= Config.YEARLY_SEASONALITY_MIN_DAYS
        params = dict(PROPHET_PARAMS)
        if not has_full_year:
            params['yearly_seasonality'] = False
        m = Prophet(**params)

        # Add Sri Lanka regressors
        for reg in EXTRA_REGRESSORS:
            if reg in train_df.columns:
                m.add_regressor(reg, mode='multiplicative')

        # Add Sri Lanka-specific yearly seasonality override (Fourier order 8) —
        # only once there's a full year to identify it against.
        if has_full_year:
            m.add_seasonality(name='yearly_slt', period=365.25, fourier_order=8)

        # Fit
        m.fit(train_df[['ds', 'y'] + [r for r in EXTRA_REGRESSORS if r in train_df.columns]])

        meta = {
            'training_rows': len(train_df),
            'date_min':      str(train_df['ds'].min().date()),
            'date_max':      str(train_df['ds'].max().date()),
            'mean_y':        round(float(train_df['y'].mean()), 2),
            'last_trained':  datetime.utcnow().isoformat() + 'Z',
        }
        return m, meta

    def _compute_in_sample_metrics(self, train_df: pd.DataFrame,
                                    model: Optional[object] = None,
                                    meta: Optional[dict] = None) -> dict:
        """
        Compute in-sample MAE, RMSE and accuracy estimate.
        Uses last 30 days as a holdout for quick evaluation.

        model/meta default to the live self._model/self._meta, but callers
        evaluating a not-yet-activated candidate must pass them explicitly.
        """
        model = model if model is not None else self._model
        meta  = meta  if meta  is not None else self._meta
        try:
            from data.feature_engineer import FeatureEngineer
            fe = FeatureEngineer()

            if len(train_df) < 60:
                return {'mae': None, 'rmse': None, 'accuracy': None}

            # Holdout: last 30 days
            holdout = train_df.tail(30).copy()
            pred_input = fe.add_regressors_to_future(
                holdout[['ds']].copy(),
                historical_mean=float(train_df['y'].mean())
            )
            forecast = model.predict(pred_input)

            actual    = holdout['y'].values
            predicted = forecast['yhat'].values[:len(actual)]

            mae  = float(np.mean(np.abs(actual - predicted)))
            rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
            mean_actual = float(np.mean(actual)) if np.mean(actual) > 0 else 1.0

            # Accuracy = 1 - MAPE (capped at 0)
            mape     = float(np.mean(np.abs((actual - predicted) / np.maximum(actual, 1)))) * 100
            accuracy = max(0.0, round(100.0 - mape, 1))

            return {
                'mae':           round(mae,  2),
                'rmse':          round(rmse, 2),
                'mape':          round(mape, 1),
                'accuracy':      accuracy,
                'trainingDays':  len(train_df),
                'lastTrained':   meta.get('last_trained'),
            }
        except Exception as exc:
            logger.warning(f"In-sample metrics error: {exc}")
            return {'mae': None, 'rmse': None, 'accuracy': None}

    def _compute_cv_metrics(self, train_df: pd.DataFrame,
                             model: Optional[object] = None,
                             meta: Optional[dict] = None) -> dict:
        """
        Full cross-validation (slower, used by /retrain endpoint).
        Uses Prophet's built-in cross_validation with rolling window.

        model/meta default to the live self._model/self._meta, but callers
        evaluating a not-yet-activated candidate must pass them explicitly.
        """
        model = model if model is not None else self._model
        meta  = meta  if meta  is not None else self._meta
        try:
            if not _PROPHET_AVAILABLE or len(train_df) < 180:
                return self._compute_in_sample_metrics(train_df, model=model, meta=meta)

            cv_results = cross_validation(
                model,
                initial='180 days',
                period='30 days',
                horizon='30 days',
                parallel=None,
            )
            pm = performance_metrics(cv_results, rolling_window=1)
            row = pm.iloc[-1]
            return {
                'mae':          round(float(row['mae']),   2),
                'rmse':         round(float(row['rmse']),  2),
                'mape':         round(float(row['mape']) * 100, 1),
                'accuracy':     round(max(0.0, 100.0 - float(row['mape']) * 100), 1),
                'trainingDays': len(train_df),
                'lastTrained':  meta.get('last_trained'),
                'horizon':      30,
            }
        except Exception as exc:
            logger.warning(f"CV metrics error: {exc}")
            return self._compute_in_sample_metrics(train_df, model=model, meta=meta)

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE — RESPONSE BUILDING
    # ─────────────────────────────────────────────────────────────────────────

    def _build_response(
        self,
        train_df: pd.DataFrame,
        raw_forecast: pd.DataFrame,
        horizon: int,
    ) -> dict:
        """Convert Prophet forecast DataFrame into the API response dict."""

        # ── Historical data (training period) ────────────────────────────────
        historical_dates = set(train_df['ds'].dt.date)
        historical = []
        for _, row in train_df.iterrows():
            historical.append({
                'ds':         row['ds'].strftime('%Y-%m-%d'),
                'y':          int(row['y']),
                'actual':     int(row['y']),
                'isForecast': False,
            })

        # ── Forecast data (future period only) ──────────────────────────────
        forecast_rows = []
        cutoff = train_df['ds'].max()
        for _, row in raw_forecast.iterrows():
            if row['ds'] > cutoff:
                forecast_rows.append({
                    'ds':         row['ds'].strftime('%Y-%m-%d'),
                    'yhat':       max(0, round(float(row['yhat']), 1)),
                    'upper':      max(0, round(float(row['yhat_upper']), 1)),
                    'lower':      max(0, round(float(row['yhat_lower']), 1)),
                    'isForecast': True,
                })

        # ── Trend analysis (compare last 30d vs previous 30d) ────────────────
        recent   = train_df.tail(30)['y'].mean()
        previous = train_df.iloc[-60:-30]['y'].mean() if len(train_df) >= 60 else recent
        trend_pct = ((recent - previous) / max(previous, 1)) * 100

        if   trend_pct >  5: trend = 'up'
        elif trend_pct < -5: trend = 'down'
        else:                trend = 'stable'

        # ── Next period total ─────────────────────────────────────────────────
        next_total = int(sum(max(0, r['yhat']) for r in forecast_rows))

        # ── Weekly average pattern ────────────────────────────────────────────
        day_names = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        train_df['dow'] = train_df['ds'].dt.dayofweek
        weekly_avg = train_df.groupby('dow')['y'].mean().round(1)
        weekly_pattern = [
            {'dayName': day_names[d], 'avgFaults': float(weekly_avg.get(d, 0))}
            for d in range(7)
        ]

        # ── Metrics ───────────────────────────────────────────────────────────
        metrics = {
            **self._meta.get('in_sample_metrics', {}),
            'trainingDays': len(train_df),
            'horizon':      horizon,
            'lastTrained':  self._meta.get('last_trained'),
        }

        return {
            'historical':      historical,
            'forecast':        forecast_rows,
            'metrics':         metrics,
            'trend':           trend,
            'trendPercent':    round(float(trend_pct), 1),
            'nextPeriodTotal': next_total,
            'weeklyPattern':   weekly_pattern,
        }

    def _insufficient_data_response(self, clean_stats: dict, horizon: int) -> dict:
        """
        SRS 5.6.2 — real (if any) historical data doesn't meet the 6-month
        minimum (Config.FORECAST_MIN_HISTORY_DAYS). Returns an explicit,
        honest state instead of silently substituting a fake-looking
        forecast: callers (app.py's predictions/dashboard handlers, and
        FR-33's resource-plan later) must check `insufficientData` before
        treating the rest of this dict as a real result. This is a distinct
        condition from `_fallback_forecast` (Prophet not installed) — the
        two calling app.py's endpoint should not be conflated, since one is
        a data-availability problem and the other is an environment/
        deployment problem.

        historical/forecast/metrics keep the SAME keys as a real forecast
        response (just empty/None) rather than omitting them, so existing
        callers that do `(d.forecast || []).map(...)` need no changes.
        """
        return {
            'insufficientData': True,
            'historyDays':      clean_stats.get('output_rows', 0),
            'requiredDays':     Config.FORECAST_MIN_HISTORY_DAYS,
            'historical':       [],
            'forecast':         [],
            'metrics': {
                'mae': None, 'rmse': None, 'accuracy': None,
                'trainingDays': clean_stats.get('output_rows', 0),
                'horizon': horizon,
                'lastTrained': None,
            },
            'trend':           None,
            'trendPercent':    None,
            'nextPeriodTotal':  None,
            'weeklyPattern':    [],
        }

    def _fallback_forecast(self, raw_df: pd.DataFrame, horizon: int) -> dict:
        """
        Simple linear extrapolation fallback when Prophet is unavailable.
        Returns the same structure as the full forecast for API compatibility.
        """
        logger.info("Using linear fallback forecast (Prophet not available)")

        if raw_df is None or raw_df.empty:
            from data.synthetic_data import SyntheticDataGenerator
            raw_df = SyntheticDataGenerator().fault_time_series(days=180)

        df = raw_df.copy()
        df['ds'] = pd.to_datetime(df['ds'])
        df = df.sort_values('ds').tail(90)

        mean_y   = float(df['y'].mean())
        recent   = float(df.tail(14)['y'].mean())
        trend    = (recent - mean_y) / max(mean_y, 1)

        historical = [
            {'ds': r['ds'].strftime('%Y-%m-%d'), 'y': int(r['y']),
             'actual': int(r['y']), 'isForecast': False}
            for _, r in df.iterrows()
        ]

        cutoff   = df['ds'].max()
        forecast = []
        for i in range(1, horizon + 1):
            day = cutoff + pd.Timedelta(days=i)
            yhat = max(0, recent * (1 + trend * (i / horizon)))
            forecast.append({
                'ds': day.strftime('%Y-%m-%d'),
                'yhat': round(yhat, 1),
                'upper': round(yhat * 1.2, 1),
                'lower': round(yhat * 0.8, 1),
                'isForecast': True,
            })

        return {
            'historical':      historical,
            'forecast':        forecast,
            'metrics':         {
                'mae': None, 'rmse': None, 'accuracy': None,
                'trainingDays': len(df), 'horizon': horizon,
                'lastTrained': datetime.utcnow().isoformat() + 'Z',
                'note': 'Linear fallback — install prophet for full model',
            },
            'trend':           'stable',
            'trendPercent':    round(trend * 100, 1),
            'nextPeriodTotal': int(sum(r['yhat'] for r in forecast)),
            'weeklyPattern':   [],
        }

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE — PERSISTENCE
    # ─────────────────────────────────────────────────────────────────────────

    def _load_saved_model(self) -> None:
        """Load the currently active version from the registry, if any."""
        active = self._registry.get_active() or self._migrate_legacy_model()
        if active is None:
            logger.info("No saved Prophet model found — will train on first request")
            return
        try:
            self._model   = self._registry.load_object(active['versionId'])
            self._meta    = active['meta']
            self._trained = True
            logger.info(
                f"Loaded saved Prophet model (version {active['versionId']}, "
                f"trained {self._meta.get('last_trained', 'unknown')})"
            )
        except Exception as exc:
            logger.warning(f"Could not load saved model: {exc}")
            self._model   = None
            self._trained = False

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
            metrics = meta.get('in_sample_metrics') or {}
            entry = self._registry.save_version(model, metrics, meta, status='active')
            logger.info(
                f"Migrated legacy prophet_model.pkl into version registry "
                f"as version {entry['versionId']}"
            )
            return entry
        except Exception as exc:
            logger.warning(f"Legacy model migration failed: {exc}")
            return None
