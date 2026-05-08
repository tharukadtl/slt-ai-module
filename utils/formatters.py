"""
utils/formatters.py — Response Formatting Utilities
====================================================
Standardises all JSON responses from the SLT AI Flask API.
Provides consistent envelope structure, data serialisation helpers,
and display-oriented formatters for numbers, dates, and geographic data.

Exports:
    success_response()     — Standard 200 envelope
    error_response()       — Standard error envelope
    paginate()             — Add pagination metadata
    fmt_lkr()              — LKR currency string
    fmt_date()             — Date string helpers
    fmt_percent()          — Percentage with 1 decimal
    fmt_km()               — Distance string
    fmt_eta()              — ETA minutes to human string
    serialise_df()         — DataFrame → JSON-safe list of dicts
    clip_series()          — Trim a list to last N items
    round_coords()         — Standardise lat/lng precision
    risk_color()           — Risk level → hex colour
    trend_arrow()          — Trend direction → arrow char + colour
    build_chart_series()   — Recharts-ready [{name, value}] data

Design:
    - All public functions are pure (no side effects, no global state)
    - datetime objects are always serialised to ISO8601 strings
    - NaN / Inf / None are normalised to null-safe defaults
    - LKR amounts are formatted for Sri Lanka locale (en-LK)
"""

import math
import logging
from datetime import datetime, date, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np

logger = logging.getLogger('slt_ai.formatters')

# ─────────────────────────────────────────────────────────────────────────────
# STANDARD RESPONSE ENVELOPES
# ─────────────────────────────────────────────────────────────────────────────

def success_response(
    data:    Any,
    message: str = 'OK',
    code:    int = 200
) -> Tuple[dict, int]:
    """
    Build a standard API success envelope.

    Args:
        data:    Serialisable payload (dict, list, scalar).
        message: Human-readable status message.
        code:    HTTP status code (default 200).

    Returns:
        (response_dict, http_code) tuple for Flask jsonify.

    Example response:
        {
          "success": true,
          "data": { ... },
          "message": "30-day forecast complete",
          "timestamp": "2026-04-20T09:15:00Z"
        }
    """
    return {
        'success':   True,
        'data':      _sanitise(data),
        'message':   str(message),
        'timestamp': _now_iso(),
    }, code


def error_response(
    message: str,
    code:    int  = 500,
    details: Any  = None
) -> Tuple[dict, int]:
    """
    Build a standard API error envelope.

    Args:
        message: Human-readable error description.
        code:    HTTP status code.
        details: Optional additional context (omitted if None).

    Returns:
        (response_dict, http_code) tuple.
    """
    body = {
        'success':   False,
        'error':     str(message),
        'timestamp': _now_iso(),
    }
    if details is not None:
        body['details'] = _sanitise(details)
    return body, code


def paginate(
    items:    List[Any],
    page:     int = 1,
    per_page: int = 20
) -> dict:
    """
    Wrap a list with pagination metadata.

    Args:
        items:    Full list (already filtered/sorted).
        page:     1-based page number.
        per_page: Items per page.

    Returns:
        {
          "items": [...],
          "page": 1,
          "perPage": 20,
          "total": 247,
          "totalPages": 13,
          "hasNext": true,
          "hasPrev": false
        }
    """
    total       = len(items)
    total_pages = max(1, math.ceil(total / per_page))
    page        = max(1, min(page, total_pages))
    start       = (page - 1) * per_page
    end         = start + per_page

    return {
        'items':      _sanitise(items[start:end]),
        'page':       page,
        'perPage':    per_page,
        'total':      total,
        'totalPages': total_pages,
        'hasNext':    page < total_pages,
        'hasPrev':    page > 1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# NUMBER FORMATTERS
# ─────────────────────────────────────────────────────────────────────────────

def fmt_lkr(
    amount: Any,
    symbol: bool = True,
    decimals: int = 2
) -> str:
    """
    Format a number as Sri Lankan Rupees (LKR).

    Args:
        amount:   Numeric value.
        symbol:   Prepend 'LKR ' (default True).
        decimals: Decimal places (default 2).

    Returns:
        Formatted string like 'LKR 12,450.00'
    """
    try:
        v    = float(amount) if amount is not None else 0.0
        sign = '-' if v < 0 else ''
        v    = abs(v)
        # Thousands-separated with comma
        fmt  = f"{v:,.{decimals}f}"
        prefix = 'LKR ' if symbol else ''
        return f"{sign}{prefix}{fmt}"
    except (ValueError, TypeError):
        return 'LKR 0.00' if symbol else '0.00'


def fmt_percent(value: Any, decimals: int = 1) -> str:
    """Format a float as a percentage string. 0.853 → '85.3%'."""
    try:
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return '—'
        # Accept both 0–1 and 0–100 inputs
        if 0 <= v <= 1:
            v *= 100
        return f"{v:.{decimals}f}%"
    except (ValueError, TypeError):
        return '—'


def fmt_km(distance_km: Any, decimals: int = 1) -> str:
    """Format a distance as kilometres string. 2.456 → '2.5 km'."""
    try:
        v = float(distance_km)
        return f"{v:.{decimals}f} km"
    except (ValueError, TypeError):
        return '— km'


def fmt_eta(minutes: Any) -> str:
    """
    Format an ETA in minutes to a human-readable string.

    Examples:
        5  → '5 min'
        70 → '1 hr 10 min'
        0  → '< 1 min'
    """
    try:
        m = int(minutes)
        if m <= 0:
            return '< 1 min'
        if m < 60:
            return f"{m} min"
        h, rem = divmod(m, 60)
        if rem == 0:
            return f"{h} hr"
        return f"{h} hr {rem} min"
    except (ValueError, TypeError):
        return '—'


def fmt_count(value: Any) -> str:
    """Format an integer count with thousands separator. 12345 → '12,345'."""
    try:
        return f"{int(value):,}"
    except (ValueError, TypeError):
        return '0'


def safe_round(value: Any, decimals: int = 2) -> Optional[float]:
    """
    Round a float safely, returning None for NaN/Inf/None.
    Useful for model metrics that may be undefined.
    """
    try:
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return None
        return round(v, decimals)
    except (ValueError, TypeError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# DATE / TIME FORMATTERS
# ─────────────────────────────────────────────────────────────────────────────

def fmt_date(
    value: Any,
    fmt: str   = '%Y-%m-%d',
    default: str = '—'
) -> str:
    """
    Convert a date/datetime/string to a formatted date string.

    Args:
        value:   date, datetime, or ISO string.
        fmt:     strftime format (default '%Y-%m-%d').
        default: Return value when input is None or invalid.
    """
    if value is None:
        return default
    try:
        if isinstance(value, (datetime, date)):
            return value.strftime(fmt)
        # Try ISO parse
        dt = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        return dt.strftime(fmt)
    except (ValueError, TypeError):
        return default


def fmt_datetime(value: Any, default: str = '—') -> str:
    """Format to 'DD Mon YYYY HH:MM' (UK style)."""
    return fmt_date(value, fmt='%d %b %Y %H:%M', default=default)


def time_ago(value: Any) -> str:
    """
    Return a human-readable relative time string.

    Examples:
        '30s ago', '5m ago', '2h ago', '3d ago'
    """
    if value is None:
        return '—'
    try:
        if isinstance(value, (datetime, date)):
            dt = datetime.combine(value, datetime.min.time()) if isinstance(value, date) else value
        else:
            dt = datetime.fromisoformat(str(value).replace('Z', '+00:00'))

        # Make naive
        if dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)

        secs = int((datetime.utcnow() - dt).total_seconds())
        if secs < 0:
            return 'just now'
        if secs < 60:
            return f"{secs}s ago"
        if secs < 3600:
            return f"{secs // 60}m ago"
        if secs < 86400:
            return f"{secs // 3600}h ago"
        return f"{secs // 86400}d ago"
    except Exception:
        return '—'


# ─────────────────────────────────────────────────────────────────────────────
# GEOGRAPHIC FORMATTERS
# ─────────────────────────────────────────────────────────────────────────────

def round_coords(lat: Any, lng: Any, decimals: int = 6) -> Tuple[Optional[float], Optional[float]]:
    """
    Round lat/lng to standard precision (6 decimal places ≈ 0.1 m accuracy).
    Returns (None, None) if input is invalid.
    """
    try:
        return round(float(lat), decimals), round(float(lng), decimals)
    except (ValueError, TypeError):
        return None, None


def coords_to_dict(lat: Any, lng: Any) -> dict:
    """Return {'lat': float, 'lng': float} dict, or {} on invalid input."""
    r_lat, r_lng = round_coords(lat, lng)
    if r_lat is None or r_lng is None:
        return {}
    return {'lat': r_lat, 'lng': r_lng}


# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY / UI FORMATTERS
# ─────────────────────────────────────────────────────────────────────────────

_RISK_COLORS = {
    'HIGH':   '#EF4444',
    'MEDIUM': '#F59E0B',
    'LOW':    '#10B981',
}

def risk_color(risk_level: str, default: str = '#6B7280') -> str:
    """Return a hex colour for a risk level (HIGH/MEDIUM/LOW)."""
    return _RISK_COLORS.get(str(risk_level).upper(), default)


_TREND_SYMBOLS = {
    'up':     ('↑', '#EF4444'),
    'down':   ('↓', '#10B981'),
    'stable': ('→', '#6B7280'),
}

def trend_arrow(trend: str) -> Tuple[str, str]:
    """
    Return (arrow_symbol, hex_color) for a trend direction.

    Args:
        trend: 'up' | 'down' | 'stable'

    Returns:
        ('↑', '#EF4444') for 'up', etc.
    """
    return _TREND_SYMBOLS.get(str(trend).lower(), ('→', '#6B7280'))


def performance_badge(score: float) -> Tuple[str, str]:
    """
    Map a KPI score (0–100) to (label, hex_color) for display badges.

    Score ranges:
        ≥90  EXCELLENT  lime green
        ≥75  GOOD       teal
        ≥60  AVERAGE    amber
        <60  NEEDS WORK red
    """
    if score >= 90:  return 'EXCELLENT',  '#AAFF00'
    if score >= 75:  return 'GOOD',       '#00E5FF'
    if score >= 60:  return 'AVERAGE',    '#FFB800'
    return 'NEEDS WORK', '#FF3060'


# ─────────────────────────────────────────────────────────────────────────────
# DATA SERIALISATION
# ─────────────────────────────────────────────────────────────────────────────

def serialise_df(
    df:           pd.DataFrame,
    date_cols:    List[str]  = None,
    round_cols:   Dict[str, int] = None,
    drop_na_cols: bool       = False
) -> List[dict]:
    """
    Convert a DataFrame to a JSON-serialisable list of dicts.
    Handles NaN, Inf, datetime, numpy types.

    Args:
        df:           Source DataFrame.
        date_cols:    Columns to format as 'YYYY-MM-DD' strings.
        round_cols:   {col_name: decimal_places} rounding map.
        drop_na_cols: Drop columns where ALL values are NaN.

    Returns:
        List of Python dicts (safe for json.dumps / Flask jsonify).
    """
    if df is None or df.empty:
        return []

    df = df.copy()

    if drop_na_cols:
        df = df.dropna(axis=1, how='all')

    if date_cols:
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                df[col] = df[col].apply(
                    lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else None
                )

    if round_cols:
        for col, dp in round_cols.items():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').round(dp)

    records = df.to_dict(orient='records')
    return [_sanitise_row(r) for r in records]


def clip_series(
    items:  List[Any],
    n:      int,
    from_end: bool = False
) -> List[Any]:
    """
    Return the first (or last) n items from a list.

    Args:
        items:    Source list.
        n:        Max items to return.
        from_end: If True, return the LAST n items.

    Returns:
        Sliced list.
    """
    if not items:
        return []
    if from_end:
        return items[-n:] if len(items) >= n else items
    return items[:n]


def build_chart_series(
    labels: List[str],
    values: List[float],
    color:  str = None
) -> List[dict]:
    """
    Build a Recharts-compatible data series.

    Args:
        labels: X-axis category names.
        values: Y-axis values (one per label).
        color:  Optional hex color for this series.

    Returns:
        [{'name': 'Mon', 'value': 22, 'color': '#2563EB'}, ...]
    """
    result = []
    for label, value in zip(labels, values):
        item = {
            'name':  str(label),
            'value': safe_round(value, 2),
        }
        if color:
            item['color'] = color
        result.append(item)
    return result


def build_time_series_chart(
    df:        pd.DataFrame,
    date_col:  str = 'ds',
    value_col: str = 'y',
    label:     str = 'value'
) -> List[dict]:
    """
    Convert a time-series DataFrame to Recharts line/area chart format.

    Returns:
        [{'date': '2026-01-01', label: 22}, ...]
    """
    if df is None or df.empty:
        return []
    result = []
    for _, row in df.iterrows():
        d = row.get(date_col)
        v = row.get(value_col)
        date_str = d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d)[:10]
        result.append({
            'date':  date_str,
            label:   safe_round(v, 1) if v is not None else None,
        })
    return result


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    """Current UTC time as ISO8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _sanitise(value: Any) -> Any:
    """
    Recursively make a value JSON-safe.
    Converts: numpy types → Python scalars, NaN/Inf → None,
    datetime → ISO string, date → YYYY-MM-DD string,
    DataFrame/Series → list of dicts / list.
    """
    if value is None:
        return None
    if isinstance(value, (bool,)):
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if math.isnan(value) or math.isinf(value):
            return None
        return float(value)
    if isinstance(value, str):
        return value
    if isinstance(value, (datetime,)):
        return value.isoformat()
    if isinstance(value, (date,)):
        return value.strftime('%Y-%m-%d')
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.DataFrame):
        return serialise_df(value)
    if isinstance(value, pd.Series):
        return [_sanitise(v) for v in value.tolist()]
    if isinstance(value, np.ndarray):
        return [_sanitise(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _sanitise(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitise(v) for v in value]
    # Fallback — try str conversion
    try:
        return str(value)
    except Exception:
        return None


def _sanitise_row(row: dict) -> dict:
    """Sanitise a single dict row (from DataFrame.to_dict)."""
    clean = {}
    for k, v in row.items():
        key = str(k)
        val = _sanitise(v)
        clean[key] = val
    return clean
