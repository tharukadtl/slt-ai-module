"""
routes/predictions.py — /api/ai/predictions Blueprint
======================================================
Prophet time-series fault volume forecasting endpoint.

Endpoints:
    GET  /api/ai/predictions              Default 30-day forecast
    GET  /api/ai/predictions?horizon=7    Short-term 7-day forecast
    GET  /api/ai/predictions?horizon=90   90-day forecast (max)
    GET  /api/ai/predictions/categories   Per-category forecasts
    POST /api/ai/predictions/retrain      Trigger Prophet retraining

Query parameters:
    horizon   (int):  Forecast horizon in days. Default 30, max 90.
    category  (str):  Filter to a single fault category (optional).
    format    (str):  'full' (default) | 'summary' (lightweight for dashboard).

Response shape:
    {
      success:         bool,
      data: {
        historical:       [{ds, y, actual, isForecast}],
        forecast:         [{ds, yhat, upper, lower, isForecast}],
        metrics:          {mae, rmse, mape, accuracy, trainingDays, horizon, lastTrained},
        trend:            "up" | "down" | "stable",
        trendPercent:     float,
        nextPeriodTotal:  int,
        weeklyPattern:    [{dayName, avgFaults}],
        dataSource:       "database" | "synthetic",
      },
      message:         str,
      timestamp:       ISO8601
    }
"""

import logging
from datetime import datetime, timezone

from flask import Blueprint, jsonify, request

from config import Config, is_db_available

logger = logging.getLogger('slt_ai.routes.predictions')

predictions_bp = Blueprint('predictions', __name__)

# ── Lazy-load model singleton (avoids circular import at module level) ─────────
_forecaster = None

def _get_forecaster():
    global _forecaster
    if _forecaster is None:
        try:
            from models.forecasting import ProphetForecaster
            _forecaster = ProphetForecaster()
        except Exception as e:
            logger.error(f"Could not load ProphetForecaster: {e}")
    return _forecaster


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _ok(data, message='OK', code=200):
    return jsonify({
        'success':   True,
        'data':      data,
        'message':   message,
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }), code


def _err(message, code=500):
    logger.warning(f"Predictions error [{code}]: {message}")
    return jsonify({
        'success':   False,
        'error':     message,
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }), code


def _parse_horizon() -> int:
    """Parse and clamp horizon query param."""
    try:
        h = int(request.args.get('horizon', Config.FORECAST_HORIZON_DAYS))
        return max(7, min(90, h))
    except (ValueError, TypeError):
        return Config.FORECAST_HORIZON_DAYS


def _load_raw_data(days_back: int = 540, category: str = None):
    """
    Load raw time-series from MySQL or fall back to synthetic data.
    Returns (DataFrame, data_source_str).
    """
    from data.data_extractor import DataExtractor
    from data.synthetic_data import SyntheticDataGenerator

    if is_db_available():
        extractor = DataExtractor()
        if category:
            # Category-filtered time-series
            cat_df = extractor.get_fault_category_breakdown(days_back=days_back)
            if cat_df is not None and not cat_df.empty:
                cat_upper = category.upper()
                filtered = cat_df[cat_df['category'] == cat_upper][['ds', 'count']].copy()
                filtered = filtered.rename(columns={'count': 'y'})
                if len(filtered) >= Config.FORECAST_MIN_HISTORY_DAYS:
                    return filtered, 'database'
        else:
            raw = extractor.get_fault_time_series(days_back=days_back)
            if raw is not None and len(raw) >= Config.FORECAST_MIN_HISTORY_DAYS:
                return raw, 'database'

    # Fallback
    logger.info("Predictions: using synthetic data")
    synth = SyntheticDataGenerator(seed=42)
    return synth.fault_time_series(days=days_back), 'synthetic'


def _to_summary(full_result: dict) -> dict:
    """
    Strip historical data from a full forecast result for lightweight responses.
    Keeps: forecast (next 7 days), metrics, trend, nextPeriodTotal.
    """
    return {
        'forecast':        full_result.get('forecast', [])[:7],
        'metrics':         full_result.get('metrics',  {}),
        'trend':           full_result.get('trend',    'stable'),
        'trendPercent':    full_result.get('trendPercent', 0.0),
        'nextPeriodTotal': full_result.get('nextPeriodTotal', 0),
        'dataSource':      full_result.get('dataSource', 'synthetic'),
    }


# ─── Routes ───────────────────────────────────────────────────────────────────

@predictions_bp.route('/predictions', methods=['GET'])
def get_predictions():
    """
    Main forecast endpoint.

    GET /api/ai/predictions
    GET /api/ai/predictions?horizon=14
    GET /api/ai/predictions?horizon=30&format=summary
    GET /api/ai/predictions?category=BROADBAND
    """
    forecaster = _get_forecaster()
    if forecaster is None:
        return _err('Prophet forecasting model not available', 503)

    horizon  = _parse_horizon()
    category = request.args.get('category', '').upper() or None
    fmt      = request.args.get('format', 'full').lower()

    # Validate category
    if category and category not in Config.FAULT_CATEGORIES + ['ALL']:
        return _err(
            f"Invalid category '{category}'. "
            f"Valid: {', '.join(Config.FAULT_CATEGORIES)}",
            400
        )

    try:
        raw_df, data_source = _load_raw_data(
            days_back=max(Config.FORECAST_MIN_HISTORY_DAYS * 2, 540),
            category=category if category != 'ALL' else None,
        )

        result = forecaster.forecast(raw_df, horizon=horizon)
        result['dataSource'] = data_source

        # Apply category label if filtered
        if category and category != 'ALL':
            result['category'] = category

        if fmt == 'summary':
            return _ok(_to_summary(result), f"{horizon}-day forecast summary")

        logger.info(
            f"Forecast served: horizon={horizon}d, "
            f"source={data_source}, rows={len(result.get('historical', []))}"
        )
        return _ok(result, f"{horizon}-day fault volume forecast")

    except Exception as exc:
        logger.exception("Unhandled forecast error")
        return _err(f"Forecast failed: {str(exc)}")


@predictions_bp.route('/predictions/categories', methods=['GET'])
def get_category_predictions():
    """
    Per-category forecasts for the next N days.

    GET /api/ai/predictions/categories?horizon=14

    Returns a forecast for each fault category (BROADBAND, FIBER, etc.)
    so the admin can see which service type is expected to spike.
    """
    forecaster = _get_forecaster()
    if forecaster is None:
        return _err('Forecasting model not available', 503)

    horizon = _parse_horizon()
    results = {}

    for category in Config.FAULT_CATEGORIES:
        try:
            raw_df, data_source = _load_raw_data(
                days_back=360,
                category=category,
            )
            forecast = forecaster.forecast(raw_df, horizon=horizon)
            results[category] = {
                'nextPeriodTotal': forecast.get('nextPeriodTotal', 0),
                'trend':           forecast.get('trend', 'stable'),
                'trendPercent':    forecast.get('trendPercent', 0.0),
                'forecast':        forecast.get('forecast', [])[:7],
                'dataSource':      data_source,
            }
        except Exception as exc:
            logger.warning(f"Category forecast failed for {category}: {exc}")
            results[category] = {'error': str(exc)}

    return _ok(
        {'categories': results, 'horizon': horizon},
        f"Per-category {horizon}-day forecast"
    )


@predictions_bp.route('/predictions/retrain', methods=['POST'])
def retrain_forecast():
    """
    Trigger a full Prophet model retrain.

    POST /api/ai/predictions/retrain
    Body: { "days_back": 540 }  (optional)

    Returns updated evaluation metrics.
    """
    forecaster = _get_forecaster()
    if forecaster is None:
        return _err('Forecasting model not available', 503)

    body      = request.get_json(silent=True) or {}
    days_back = int(body.get('days_back', 540))
    days_back = max(90, min(730, days_back))

    try:
        raw_df, data_source = _load_raw_data(days_back=days_back)
        metrics = forecaster.retrain(raw_df)
        metrics['dataSource'] = data_source

        logger.info(f"Prophet retrained — MAE={metrics.get('mae')}, acc={metrics.get('accuracy')}%")
        return _ok(metrics, "Prophet model retrained successfully")

    except Exception as exc:
        logger.exception("Retrain error")
        return _err(f"Retrain failed: {str(exc)}")


@predictions_bp.route('/predictions/history', methods=['GET'])
def get_historical():
    """
    Raw historical fault counts only (no forecast).
    Useful for the dashboard trend chart.

    GET /api/ai/predictions/history?days=90
    """
    try:
        days   = int(request.args.get('days', 90))
        days   = max(7, min(365, days))
        raw_df, data_source = _load_raw_data(days_back=days)

        if raw_df is None or raw_df.empty:
            return _ok({'historical': [], 'dataSource': 'none'}, 'No data')

        from data.data_cleaner import DataCleaner
        cleaner  = DataCleaner()
        clean_df, _ = cleaner.clean_time_series(raw_df, min_rows=1)
        if clean_df is None:
            clean_df = raw_df

        historical = [
            {
                'ds':    row['ds'].strftime('%Y-%m-%d'),
                'y':     int(row['y']),
                'actual': int(row['y']),
            }
            for _, row in clean_df.iterrows()
        ]

        return _ok(
            {'historical': historical, 'totalDays': len(historical), 'dataSource': data_source},
            f"{len(historical)} days of fault history"
        )

    except Exception as exc:
        logger.exception("History fetch error")
        return _err(f"History fetch failed: {str(exc)}")
