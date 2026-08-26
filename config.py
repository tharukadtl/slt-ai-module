"""
config.py — SLT AI Module Configuration
========================================
Loads all settings from environment variables (.env file).
Provides database engine, logging setup, and shared constants.

Usage:
    from config import Config, get_db_engine, logger
"""

import os
import logging
import sys
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

# ─── Load .env ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / '.env')


# ─── Logging setup ────────────────────────────────────────────────────────────
def setup_logging() -> logging.Logger:
    """Configure structured logging to both console and file."""
    log_level = getattr(logging, os.getenv('LOG_LEVEL', 'INFO').upper(), logging.INFO)
    log_file  = os.getenv('LOG_FILE', './logs/ai_module.log')

    # Ensure log directory exists
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    root = logging.getLogger()
    root.setLevel(log_level)

    # Console handler — reconfigure to UTF-8 so emoji in log messages don't
    # crash logging on Windows' default cp1252 console encoding.
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(log_level)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    # File handler
    try:
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        root.addHandler(fh)
    except (IOError, OSError) as e:
        root.warning(f"Could not open log file {log_file}: {e}")

    return logging.getLogger('slt_ai')


logger = setup_logging()


# ─── Config class ─────────────────────────────────────────────────────────────
class Config:
    """Central configuration — all values come from environment / .env."""

    # Flask
    DEBUG       : bool  = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    HOST        : str   = os.getenv('FLASK_HOST', '0.0.0.0')
    PORT        : int   = int(os.getenv('FLASK_PORT', 5000))
    SECRET_KEY  : str   = os.getenv('SECRET_KEY', 'dev-only-change-me')

    # Database
    DB_HOST     : str   = os.getenv('DB_HOST', 'localhost')
    DB_PORT     : int   = int(os.getenv('DB_PORT', 3306))
    DB_NAME     : str   = os.getenv('DB_NAME', 'slt_fieldops_db')
    DB_USER     : str   = os.getenv('DB_USER', 'root')
    DB_PASSWORD : str   = os.getenv('DB_PASSWORD', '1234')
    DB_POOL_SIZE: int   = int(os.getenv('DB_POOL_SIZE', 5))
    DB_POOL_RECYCLE: int= int(os.getenv('DB_POOL_RECYCLE', 3600))

    @classmethod
    def db_url(cls) -> str:
        """Build SQLAlchemy connection URL."""
        return (
            f"mysql+pymysql://{cls.DB_USER}:{cls.DB_PASSWORD}"
            f"@{cls.DB_HOST}:{cls.DB_PORT}/{cls.DB_NAME}"
            f"?charset=utf8mb4"
        )

    # Spring Boot API (fallback)
    SPRING_API_URL  : str = os.getenv('SPRING_API_URL', 'http://localhost:8080')
    SPRING_API_TOKEN: str = os.getenv('SPRING_API_TOKEN', '')

    # AI model settings
    MODEL_DIR               : str   = os.getenv('MODEL_DIR', './models/saved')
    FORECAST_HORIZON_DAYS   : int   = int(os.getenv('FORECAST_HORIZON_DAYS', 30))
    # Spec §5.6.1: fall back to synthetic data when real history is < 6 months
    FORECAST_MIN_HISTORY_DAYS: int  = int(os.getenv('FORECAST_MIN_HISTORY_DAYS', 180))
    # A distinct, larger threshold from FORECAST_MIN_HISTORY_DAYS above: that one
    # gates whether forecasting runs at all, this one gates whether an annual
    # seasonal component is even identifiable from the training window. An 8-term
    # yearly Fourier series fit on under a year of data is unidentifiable and
    # extrapolates away (QA_Compliance_Consolidated_Report.md, AI-003) — 365 is
    # one full cycle, the minimum below which Prophet has never actually observed
    # a full year to fit a yearly pattern against.
    YEARLY_SEASONALITY_MIN_DAYS: int = int(os.getenv('YEARLY_SEASONALITY_MIN_DAYS', 365))
    KMEANS_N_CLUSTERS       : int   = int(os.getenv('KMEANS_N_CLUSTERS', 5))
    KMEANS_RANDOM_STATE     : int   = int(os.getenv('KMEANS_RANDOM_STATE', 42))
    ROUTE_SEARCH_RADIUS_KM  : float = float(os.getenv('ROUTE_SEARCH_RADIUS_KM', 50))
    # Average travel speed assumption for shortest-path ETA (FR-29, SRS 5.6.6).
    # Matches fieldops' LocationService ETA assumption (30 km/h) for consistency.
    ROUTE_AVG_SPEED_KMH     : float = float(os.getenv('ROUTE_AVG_SPEED_KMH', 30))
    RETRAIN_INTERVAL_HOURS  : int   = int(os.getenv('RETRAIN_INTERVAL_HOURS', 24))

    # Sri Lanka geographic bounds
    SL_LAT_MIN   : float = float(os.getenv('SL_LAT_MIN',  5.9))
    SL_LAT_MAX   : float = float(os.getenv('SL_LAT_MAX',  9.9))
    SL_LNG_MIN   : float = float(os.getenv('SL_LNG_MIN',  79.5))
    SL_LNG_MAX   : float = float(os.getenv('SL_LNG_MAX',  81.9))
    SL_CENTER_LAT: float = float(os.getenv('SL_CENTER_LAT', 7.8731))
    SL_CENTER_LNG: float = float(os.getenv('SL_CENTER_LNG', 80.7718))

    # Sri Lanka district reference points (used as cluster seeds + labels)
    SL_DISTRICTS: list = [
        {'name': 'Colombo',     'lat': 6.9271, 'lng': 79.8612},
        {'name': 'Kandy',       'lat': 7.2906, 'lng': 80.6337},
        {'name': 'Galle',       'lat': 6.0535, 'lng': 80.2210},
        {'name': 'Jaffna',      'lat': 9.6615, 'lng': 80.0255},
        {'name': 'Batticaloa',  'lat': 7.7170, 'lng': 81.6924},
        {'name': 'Anuradhapura','lat': 8.3114, 'lng': 80.4037},
        {'name': 'Trincomalee', 'lat': 8.5874, 'lng': 81.2152},
        {'name': 'Kurunegala',  'lat': 7.4818, 'lng': 80.3609},
        {'name': 'Ratnapura',   'lat': 6.6828, 'lng': 80.3992},
    ]

    # Fault categories
    FAULT_CATEGORIES: list = [
        'BROADBAND', 'FIBER', 'TELEPHONE', 'TELEVISION', 'OTHER'
    ]

    # Valid fault statuses
    FAULT_STATUSES: list = [
        'OPEN', 'ASSIGNED', 'IN_PROGRESS', 'TRAVELLING',
        'COMPLETED', 'CANCELLED'
    ]

    # Technician availability statuses
    TECH_AVAILABLE_STATUSES: list = ['AVAILABLE', 'ACCEPTED']


# ─── Database engine singleton ────────────────────────────────────────────────
_engine = None


def get_db_engine():
    """
    Return a shared SQLAlchemy engine (singleton).
    Raises RuntimeError if connection fails — caller should handle gracefully.
    """
    global _engine
    if _engine is not None:
        return _engine

    try:
        _engine = create_engine(
            Config.db_url(),
            poolclass=QueuePool,
            pool_size=Config.DB_POOL_SIZE,
            pool_recycle=Config.DB_POOL_RECYCLE,
            pool_pre_ping=True,          # detect stale connections
            connect_args={
                'connect_timeout': 10,
                'charset': 'utf8mb4',
            }
        )
        # Test connection
        with _engine.connect() as conn:
            conn.execute(__import__('sqlalchemy').text('SELECT 1'))
        logger.info(
            f"✅ Database connected: {Config.DB_HOST}:{Config.DB_PORT}/{Config.DB_NAME}"
        )
        return _engine

    except Exception as exc:
        logger.warning(f"⚠️  Database not available: {exc}")
        logger.info("   AI module will use synthetic data as fallback.")
        _engine = None
        return None


def is_db_available() -> bool:
    """Quick check without side effects."""
    engine = get_db_engine()
    if engine is None:
        return False
    try:
        with engine.connect() as conn:
            conn.execute(__import__('sqlalchemy').text('SELECT 1'))
        return True
    except Exception:
        return False


def has_sufficient_history(row_count: int, min_days: int = None) -> bool:
    """
    SRS 5.6.2 (and 5.6.8 FR-33, which reuses the same rule) — single source
    of truth for whether `row_count` days of historical data meets the
    minimum required for a real forecast, instead of a synthetic/fallback
    substitute.

    Previously this comparison was duplicated inline in three places (app.py's
    /api/ai/predictions and /api/ai/dashboard handlers checking a raw fetched
    row count, and data_cleaner.clean_time_series checking a post-cleaning
    row count) — which could disagree with each other for the same underlying
    data, since cleaning can drop rows. There is now exactly one place this
    ">=" comparison is made.

    Args:
        row_count: Number of days of historical data available.
        min_days:  Override threshold (e.g. a CSV-retraining path that uses a
                   deliberately lower bar than the live-forecast minimum).
                   Defaults to Config.FORECAST_MIN_HISTORY_DAYS.
    """
    threshold = Config.FORECAST_MIN_HISTORY_DAYS if min_days is None else min_days
    return row_count >= threshold
