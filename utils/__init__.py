"""
utils/__init__.py
=================
Makes the utils/ directory a Python package.

Re-exports the most-used public helpers so callers can write:

    from utils import success_response, error_response
    from utils import validate_coords, validate_horizon
    from utils import DBConnector, execute_query

instead of the longer dotted paths.
"""

# ── db_connector ──────────────────────────────────────────────────────────────
from utils.db_connector import (
    DBConnector,
    get_engine,
    db_available,
    reset_engine,
    execute_query,
    execute_write,
)

# ── validators ────────────────────────────────────────────────────────────────
from utils.validators import (
    validate_int,
    validate_float,
    validate_latitude,
    validate_longitude,
    validate_coords,
    validate_horizon,
    validate_n_clusters,
    validate_days_back,
    validate_limit,
    validate_radius_km,
    validate_category,
    validate_fault_status,
    validate_period,
    validate_response_format,
    validate_description,
    validate_date_string,
    validate_date_range,
    validate_fault_list,
    validate_model_list,
    validate_prediction_request,
    validate_cluster_request,
    validate_route_request,
    validate_shortest_path_request,
    validate_classify_request,
)

# ── formatters ────────────────────────────────────────────────────────────────
from utils.formatters import (
    success_response,
    error_response,
    paginate,
    fmt_lkr,
    fmt_percent,
    fmt_km,
    fmt_eta,
    fmt_count,
    fmt_date,
    fmt_datetime,
    time_ago,
    safe_round,
    round_coords,
    coords_to_dict,
    risk_color,
    trend_arrow,
    performance_badge,
    serialise_df,
    clip_series,
    build_chart_series,
    build_time_series_chart,
)

__all__ = [
    # db_connector
    "DBConnector", "get_engine", "db_available", "reset_engine",
    "execute_query", "execute_write",
    # validators
    "validate_int", "validate_float",
    "validate_latitude", "validate_longitude", "validate_coords",
    "validate_horizon", "validate_n_clusters", "validate_days_back",
    "validate_limit", "validate_radius_km",
    "validate_category", "validate_fault_status", "validate_period",
    "validate_response_format", "validate_description",
    "validate_date_string", "validate_date_range",
    "validate_fault_list", "validate_model_list",
    "validate_prediction_request", "validate_cluster_request",
    "validate_route_request", "validate_shortest_path_request", "validate_classify_request",
    # formatters
    "success_response", "error_response", "paginate",
    "fmt_lkr", "fmt_percent", "fmt_km", "fmt_eta", "fmt_count",
    "fmt_date", "fmt_datetime", "time_ago", "safe_round",
    "round_coords", "coords_to_dict",
    "risk_color", "trend_arrow", "performance_badge",
    "serialise_df", "clip_series",
    "build_chart_series", "build_time_series_chart",
]
