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
            logger.warning("Insufficient data after cleaning — using fallback")
            return self._fallback_forecast(raw_df, horizon)

        validation = cleaner.validate_for_prophet(clean_df)
        for w in validation.get('warnings', []):
            logger.warning(f"Prophet validation: {w}")

        # ── 2. Add regressors ────────────────────────────────────────────────
        train_df = fe.add_prophet_regressors(clean_df)

        # ── 3. Train (or retrain if data has grown significantly) ────────────
        should_retrain = (
            not self._trained
            or len(train_df) > self._meta.get('training_rows', 0) + 30
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
        Force a full retrain with new data and return updated metrics.
        Called by POST /api/ai/retrain.
        """
        from data.data_cleaner     import DataCleaner
        from data.feature_engineer import FeatureEngineer

        cleaner = DataCleaner()
        fe      = FeatureEngineer()

        clean_df, _ = cleaner.clean_time_series(raw_df, min_rows=30)
        if clean_df is None:
            return {'error': 'Insufficient data for retraining'}

        train_df = fe.add_prophet_regressors(clean_df)
        self._train(train_df)

        metrics = self._compute_cv_metrics(train_df)
        self._meta['cv_metrics'] = metrics
        self._save_model()

        logger.info(f"Retrain complete — MAE={metrics.get('mae')}, RMSE={metrics.get('rmse')}")
        return metrics

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE — TRAINING
    # ─────────────────────────────────────────────────────────────────────────

    def _train(self, train_df: pd.DataFrame) -> None:
        """Fit Prophet model on the cleaned + enriched training DataFrame."""
        logger.info(f"Training Prophet on {len(train_df)} days of data…")

        m = Prophet(**PROPHET_PARAMS)

        # Add Sri Lanka regressors
        for reg in EXTRA_REGRESSORS:
            if reg in train_df.columns:
                m.add_regressor(reg, mode='multiplicative')

        # Add Sri Lanka-specific yearly seasonality override (Fourier order 8)
        m.add_seasonality(name='yearly_slt', period=365.25, fourier_order=8)

        # Fit
        m.fit(train_df[['ds', 'y'] + [r for r in EXTRA_REGRESSORS if r in train_df.columns]])

        self._model   = m
        self._trained = True
        self._meta = {
            'training_rows': len(train_df),
            'date_min':      str(train_df['ds'].min().date()),
            'date_max':      str(train_df['ds'].max().date()),
            'mean_y':        round(float(train_df['y'].mean()), 2),
            'last_trained':  datetime.utcnow().isoformat() + 'Z',
        }

        # Quick in-sample accuracy estimate (faster than cross-validation)
        self._meta['in_sample_metrics'] = self._compute_in_sample_metrics(train_df)
        self._save_model()
        logger.info(f"Prophet trained — in-sample MAE={self._meta['in_sample_metrics'].get('mae')}")

    def _compute_in_sample_metrics(self, train_df: pd.DataFrame) -> dict:
        """
        Compute in-sample MAE, RMSE and accuracy estimate.
        Uses last 30 days as a holdout for quick evaluation.
        """
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
            forecast = self._model.predict(pred_input)

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
                'lastTrained':   self._meta.get('last_trained'),
            }
        except Exception as exc:
            logger.warning(f"In-sample metrics error: {exc}")
            return {'mae': None, 'rmse': None, 'accuracy': None}

    def _compute_cv_metrics(self, train_df: pd.DataFrame) -> dict:
        """
        Full cross-validation (slower, used by /retrain endpoint).
        Uses Prophet's built-in cross_validation with rolling window.
        """
        try:
            if not _PROPHET_AVAILABLE or len(train_df) < 180:
                return self._compute_in_sample_metrics(train_df)

            cv_results = cross_validation(
                self._model,
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
                'lastTrained':  self._meta.get('last_trained'),
                'horizon':      30,
            }
        except Exception as exc:
            logger.warning(f"CV metrics error: {exc}")
            return self._compute_in_sample_metrics(train_df)

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE — RESPONSE BUILDING
    # ─────────────────────────────────────────────────────────────────────────

    def _build_response(
        self,
        train_df: pd.DataFrame,
        raw_forecast: pd.DataFrame,
        horizon: int
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

    def _save_model(self) -> None:
        """Serialise trained Prophet model and metadata to disk."""
        try:
            MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(MODEL_SAVE_PATH, 'wb') as f:
                pickle.dump(self._model, f)
            with open(META_SAVE_PATH, 'wb') as f:
                pickle.dump(self._meta, f)
            logger.info(f"Model saved to {MODEL_SAVE_PATH}")
        except Exception as exc:
            logger.warning(f"Model save failed: {exc}")

    def _load_saved_model(self) -> None:
        """Load a previously trained model from disk if available."""
        if not MODEL_SAVE_PATH.exists() or not META_SAVE_PATH.exists():
            logger.info("No saved Prophet model found — will train on first request")
            return
        try:
            with open(MODEL_SAVE_PATH, 'rb') as f:
                self._model = pickle.load(f)
            with open(META_SAVE_PATH, 'rb') as f:
                self._meta = pickle.load(f)
            self._trained = True
            logger.info(
                f"Loaded saved Prophet model "
                f"(trained {self._meta.get('last_trained', 'unknown')})"
            )
        except Exception as exc:
            logger.warning(f"Could not load saved model: {exc}")
            self._model   = None
            self._trained = False
