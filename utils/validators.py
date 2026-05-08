"""
utils/validators.py — Input Validation
=======================================
Centralised validation utilities for all Flask route inputs.
All validators return a (value, error_message) tuple:
    - On success: (validated_value, None)
    - On failure: (None, error_string)

Covers:
    - GPS coordinates (Sri Lanka bounds)
    - Date ranges and periods
    - Integer / float query params with range clamping
    - Forecast horizon
    - Fault category and status enums
    - Fault description text (XSS-safe sanitisation)
    - Batch request lists
    - Full request body validation (composite validators)

Usage:
    from utils.validators import validate_coords, validate_horizon

    lat, err = validate_coords_lat(request.args.get('lat'))
    if err:
        return _err(err, 400)

    horizon, err = validate_horizon(request.args.get('horizon'))
"""

import re
import logging
import html
from datetime import datetime, date
from typing import Any, List, Optional, Tuple

from config import Config

logger = logging.getLogger('slt_ai.validators')

# ── Type alias for validator return ──────────────────────────────────────────
ValidResult = Tuple[Any, Optional[str]]   # (value_or_None, error_or_None)


# ─────────────────────────────────────────────────────────────────────────────
# NUMERIC VALIDATORS
# ─────────────────────────────────────────────────────────────────────────────

def validate_int(
    value: Any,
    name: str     = 'value',
    minimum: int  = None,
    maximum: int  = None,
    default: int  = None,
    required: bool = False
) -> ValidResult:
    """
    Validate an integer parameter.

    Args:
        value:    Raw input (string from query params, etc.)
        name:     Parameter name (used in error messages).
        minimum:  Minimum allowed value (inclusive).
        maximum:  Maximum allowed value (inclusive).
        default:  Value to return when input is None/empty and not required.
        required: If True, return error when value is missing.

    Returns:
        (int_value, None) on success, (None, error_str) on failure.
    """
    if value is None or str(value).strip() == '':
        if required:
            return None, f"'{name}' is required"
        if default is not None:
            return default, None
        return None, None

    try:
        v = int(str(value).strip())
    except (ValueError, TypeError):
        return None, f"'{name}' must be an integer, got: {value!r}"

    if minimum is not None and v < minimum:
        return None, f"'{name}' must be ≥ {minimum}, got {v}"
    if maximum is not None and v > maximum:
        return None, f"'{name}' must be ≤ {maximum}, got {v}"

    return v, None


def validate_float(
    value: Any,
    name: str       = 'value',
    minimum: float  = None,
    maximum: float  = None,
    default: float  = None,
    required: bool  = False,
    decimals: int   = None
) -> ValidResult:
    """
    Validate a float parameter.

    Args:
        decimals: If set, round to this many decimal places.
    """
    if value is None or str(value).strip() == '':
        if required:
            return None, f"'{name}' is required"
        if default is not None:
            return default, None
        return None, None

    try:
        v = float(str(value).strip())
    except (ValueError, TypeError):
        return None, f"'{name}' must be a number, got: {value!r}"

    if minimum is not None and v < minimum:
        return None, f"'{name}' must be ≥ {minimum}, got {v}"
    if maximum is not None and v > maximum:
        return None, f"'{name}' must be ≤ {maximum}, got {v}"

    if decimals is not None:
        v = round(v, decimals)

    return v, None


# ─────────────────────────────────────────────────────────────────────────────
# GPS VALIDATORS
# ─────────────────────────────────────────────────────────────────────────────

def validate_latitude(value: Any, required: bool = True) -> ValidResult:
    """Validate a latitude value within Sri Lanka bounds."""
    lat, err = validate_float(
        value, name='lat', required=required,
        minimum=Config.SL_LAT_MIN, maximum=Config.SL_LAT_MAX,
        decimals=6
    )
    if err:
        return None, (
            f"{err}. "
            f"Sri Lanka latitude must be between "
            f"{Config.SL_LAT_MIN} and {Config.SL_LAT_MAX}."
        )
    return lat, None


def validate_longitude(value: Any, required: bool = True) -> ValidResult:
    """Validate a longitude value within Sri Lanka bounds."""
    lng, err = validate_float(
        value, name='lng', required=required,
        minimum=Config.SL_LNG_MIN, maximum=Config.SL_LNG_MAX,
        decimals=6
    )
    if err:
        return None, (
            f"{err}. "
            f"Sri Lanka longitude must be between "
            f"{Config.SL_LNG_MIN} and {Config.SL_LNG_MAX}."
        )
    return lng, None


def validate_coords(lat_raw: Any, lng_raw: Any) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """
    Validate both lat and lng together.

    Returns:
        (lat, lng, None) on success, (None, None, error_str) on failure.
    """
    lat, err = validate_latitude(lat_raw, required=True)
    if err:
        return None, None, err

    lng, err = validate_longitude(lng_raw, required=True)
    if err:
        return None, None, err

    return lat, lng, None


# ─────────────────────────────────────────────────────────────────────────────
# MODEL PARAMETER VALIDATORS
# ─────────────────────────────────────────────────────────────────────────────

def validate_horizon(value: Any) -> ValidResult:
    """
    Validate a forecast horizon (days).
    Valid range: 7–90 days. Default: Config.FORECAST_HORIZON_DAYS.
    """
    h, err = validate_int(
        value, name='horizon',
        minimum=7, maximum=90,
        default=Config.FORECAST_HORIZON_DAYS
    )
    if err:
        return None, f"{err}. Valid range: 7–90 days."
    return h, None


def validate_n_clusters(value: Any) -> ValidResult:
    """Validate K-Means cluster count. Valid range: 2–10. Default: 5."""
    n, err = validate_int(
        value, name='n_clusters',
        minimum=2, maximum=10,
        default=Config.KMEANS_N_CLUSTERS
    )
    if err:
        return None, f"{err}. Valid range: 2–10."
    return n, None


def validate_days_back(value: Any, minimum: int = 7, maximum: int = 730) -> ValidResult:
    """Validate a history window in days."""
    d, err = validate_int(
        value, name='days',
        minimum=minimum, maximum=maximum,
        default=180
    )
    if err:
        return None, f"{err}. Valid range: {minimum}–{maximum} days."
    return d, None


def validate_limit(value: Any, max_limit: int = 20) -> ValidResult:
    """Validate a result limit (pagination / top-k)."""
    lim, err = validate_int(
        value, name='limit',
        minimum=1, maximum=max_limit,
        default=5
    )
    if err:
        return None, f"{err}. Valid range: 1–{max_limit}."
    return lim, None


def validate_radius_km(value: Any) -> ValidResult:
    """Validate a search radius in kilometres."""
    r, err = validate_float(
        value, name='radius',
        minimum=1.0, maximum=200.0,
        default=Config.ROUTE_SEARCH_RADIUS_KM,
        decimals=1
    )
    if err:
        return None, f"{err}. Valid range: 1.0–200.0 km."
    return r, None


# ─────────────────────────────────────────────────────────────────────────────
# ENUM / STRING VALIDATORS
# ─────────────────────────────────────────────────────────────────────────────

def validate_category(value: Any, required: bool = False) -> ValidResult:
    """
    Validate a fault category string.
    Valid values: BROADBAND, FIBER, TELEPHONE, TELEVISION, OTHER, ALL.
    """
    if value is None or str(value).strip() == '':
        if required:
            return None, f"'category' is required. Valid: {', '.join(Config.FAULT_CATEGORIES)}"
        return None, None   # optional — caller treats None as "all categories"

    v = str(value).strip().upper()
    valid = set(Config.FAULT_CATEGORIES) | {'ALL'}
    if v not in valid:
        return None, (
            f"Invalid category '{v}'. "
            f"Valid values: {', '.join(sorted(valid))}"
        )
    return v if v != 'ALL' else None, None


def validate_fault_status(value: Any, required: bool = False) -> ValidResult:
    """Validate a fault status string."""
    if value is None or str(value).strip() == '':
        if required:
            return None, f"'status' is required. Valid: {', '.join(Config.FAULT_STATUSES)}"
        return None, None

    v = str(value).strip().upper()
    if v not in Config.FAULT_STATUSES and v != 'ALL':
        return None, (
            f"Invalid status '{v}'. "
            f"Valid: {', '.join(Config.FAULT_STATUSES + ['ALL'])}"
        )
    return v if v != 'ALL' else None, None


def validate_period(value: Any) -> ValidResult:
    """Validate a KPI period string."""
    valid = {'DAILY', 'WEEKLY', 'MONTHLY'}
    if value is None or str(value).strip() == '':
        return 'MONTHLY', None
    v = str(value).strip().upper()
    if v not in valid:
        return None, f"Invalid period '{v}'. Valid: DAILY, WEEKLY, MONTHLY"
    return v, None


def validate_response_format(value: Any) -> ValidResult:
    """Validate a response format param ('full' or 'summary')."""
    valid = {'full', 'summary'}
    if value is None:
        return 'full', None
    v = str(value).strip().lower()
    if v not in valid:
        return None, f"Invalid format '{v}'. Valid: full, summary"
    return v, None


# ─────────────────────────────────────────────────────────────────────────────
# TEXT VALIDATORS
# ─────────────────────────────────────────────────────────────────────────────

def validate_description(
    value: Any,
    required: bool    = True,
    min_len: int      = 3,
    max_len: int      = 1000,
    sanitise: bool    = True
) -> ValidResult:
    """
    Validate and sanitise a fault description text field.

    Sanitisation:
        - HTML-escape special characters (prevents XSS if ever rendered)
        - Strip leading/trailing whitespace
        - Collapse multiple spaces to single space
        - Remove non-printable control characters

    Args:
        value:    Raw input string.
        required: Return error if empty.
        min_len:  Minimum length after sanitisation.
        max_len:  Maximum length.
        sanitise: Whether to apply XSS sanitisation.

    Returns:
        (clean_str, None) or (None, error_str).
    """
    if value is None or str(value).strip() == '':
        if required:
            return None, f"'description' is required (min {min_len} characters)"
        return '', None

    v = str(value)

    # Remove control characters (keep printable + newlines)
    v = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', v)

    # Collapse whitespace
    v = re.sub(r'[ \t]+', ' ', v).strip()

    if sanitise:
        v = html.escape(v, quote=False)

    if len(v) < min_len:
        return None, f"'description' too short (min {min_len} chars, got {len(v)})"
    if len(v) > max_len:
        return None, f"'description' too long (max {max_len} chars, got {len(v)})"

    return v, None


# ─────────────────────────────────────────────────────────────────────────────
# DATE VALIDATORS
# ─────────────────────────────────────────────────────────────────────────────

def validate_date_string(
    value: Any,
    name: str      = 'date',
    required: bool = False,
    fmt: str       = '%Y-%m-%d'
) -> ValidResult:
    """
    Validate a date string in YYYY-MM-DD format.

    Returns:
        (datetime.date, None) or (None, error_str).
    """
    if value is None or str(value).strip() == '':
        if required:
            return None, f"'{name}' is required (format: YYYY-MM-DD)"
        return None, None

    try:
        return datetime.strptime(str(value).strip(), fmt).date(), None
    except ValueError:
        return None, f"'{name}' must be in {fmt} format, got: {value!r}"


def validate_date_range(
    start_raw: Any,
    end_raw: Any,
    max_days: int = 730
) -> Tuple[Optional[date], Optional[date], Optional[str]]:
    """
    Validate a start + end date range.

    Rules:
        - start must be before end
        - range must not exceed max_days

    Returns:
        (start_date, end_date, None) or (None, None, error_str).
    """
    start, err = validate_date_string(start_raw, name='start_date')
    if err:
        return None, None, err

    end, err = validate_date_string(end_raw, name='end_date')
    if err:
        return None, None, err

    if start is None or end is None:
        return start, end, None

    if start > end:
        return None, None, "'start_date' must be before 'end_date'"

    span = (end - start).days
    if span > max_days:
        return None, None, f"Date range too large: {span} days (max {max_days})"

    return start, end, None


# ─────────────────────────────────────────────────────────────────────────────
# BATCH / LIST VALIDATORS
# ─────────────────────────────────────────────────────────────────────────────

def validate_fault_list(
    faults: Any,
    max_items: int = 50
) -> Tuple[Optional[List[dict]], Optional[str]]:
    """
    Validate the 'faults' array in a batch-assign request.

    Each item must have: lat (float), lng (float).
    Optional: fault_id (int), priority (str).

    Returns:
        (validated_list, None) or (None, error_str).
    """
    if not isinstance(faults, list):
        return None, "'faults' must be a JSON array"
    if len(faults) == 0:
        return None, "'faults' array is empty"
    if len(faults) > max_items:
        return None, f"Too many faults: {len(faults)} (max {max_items})"

    valid_items  = []
    errors       = []

    for i, item in enumerate(faults):
        if not isinstance(item, dict):
            errors.append(f"Item {i}: must be an object, got {type(item).__name__}")
            continue

        lat, lng, coord_err = validate_coords(item.get('lat'), item.get('lng'))
        if coord_err:
            errors.append(f"Item {i}: {coord_err}")
            continue

        priority = str(item.get('priority', 'MEDIUM')).upper()
        if priority not in ('HIGH', 'MEDIUM', 'LOW'):
            priority = 'MEDIUM'

        valid_items.append({
            'fault_id': item.get('fault_id', i + 1),
            'lat':      lat,
            'lng':      lng,
            'priority': priority,
        })

    if errors:
        return None, f"Validation errors: {'; '.join(errors[:5])}"

    return valid_items, None


def validate_model_list(
    models: Any,
    valid_models: tuple = ('forecaster', 'clusterer', 'classifier', 'router')
) -> ValidResult:
    """
    Validate the 'models' array in a retrain request.
    Returns (list_of_model_names, None) or (None, error_str).
    """
    if models is None:
        return list(valid_models), None

    if not isinstance(models, list):
        return None, "'models' must be a JSON array"

    validated = []
    unknown   = []
    for m in models:
        m_str = str(m).strip().lower()
        if m_str in valid_models:
            validated.append(m_str)
        else:
            unknown.append(m_str)

    if unknown:
        return None, (
            f"Unknown model(s): {', '.join(unknown)}. "
            f"Valid: {', '.join(valid_models)}"
        )
    if not validated:
        return None, "'models' array is empty"

    return validated, None


# ─────────────────────────────────────────────────────────────────────────────
# COMPOSITE VALIDATORS (validate an entire request in one call)
# ─────────────────────────────────────────────────────────────────────────────

def validate_prediction_request(args: dict) -> Tuple[dict, Optional[str]]:
    """
    Validate all parameters for GET /api/ai/predictions.

    Args:
        args: Flask request.args dict.

    Returns:
        ({'horizon': int, 'category': str|None, 'format': str}, None)
        or ({}, error_str)
    """
    horizon, err = validate_horizon(args.get('horizon'))
    if err:
        return {}, err

    category, err = validate_category(args.get('category'))
    if err:
        return {}, err

    fmt, err = validate_response_format(args.get('format'))
    if err:
        return {}, err

    days_back, err = validate_days_back(
        args.get('days_back'), minimum=90, maximum=730
    )
    if err:
        return {}, err

    return {
        'horizon':   horizon,
        'category':  category,
        'format':    fmt,
        'days_back': days_back or 540,
    }, None


def validate_cluster_request(args: dict) -> Tuple[dict, Optional[str]]:
    """
    Validate all parameters for GET /api/ai/clusters.
    """
    n, err = validate_n_clusters(args.get('n_clusters'))
    if err:
        return {}, err

    days, err = validate_days_back(args.get('days'))
    if err:
        return {}, err

    category, err = validate_category(args.get('category'))
    if err:
        return {}, err

    return {'n_clusters': n, 'days': days, 'category': category}, None


def validate_route_request(args: dict) -> Tuple[dict, Optional[str]]:
    """
    Validate all parameters for GET /api/ai/optimize-route.
    """
    lat, lng, err = validate_coords(args.get('lat'), args.get('lng'))
    if err:
        return {}, err

    limit, err = validate_limit(args.get('limit'), max_limit=20)
    if err:
        return {}, err

    available_only = str(args.get('available_only', 'false')).lower() == 'true'

    return {
        'lat':            lat,
        'lng':            lng,
        'limit':          limit,
        'available_only': available_only,
    }, None


def validate_classify_request(body: dict) -> Tuple[dict, Optional[str]]:
    """
    Validate request body for POST /api/ai/dashboard/classify.
    """
    description, err = validate_description(
        body.get('description'), required=True, min_len=3, max_len=1000
    )
    if err:
        return {}, err

    context = body.get('context', {})
    if not isinstance(context, dict):
        context = {}

    # Validate hour if provided
    hour = context.get('hour')
    if hour is not None:
        h, err = validate_int(hour, name='context.hour', minimum=0, maximum=23)
        if err:
            return {}, err
        context['hour'] = h

    return {'description': description, 'context': context}, None
