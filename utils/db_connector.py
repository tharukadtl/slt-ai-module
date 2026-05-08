"""
utils/db_connector.py — MySQL Connection Manager
=================================================
Centralised database connection utilities for the SLT AI module.
Wraps SQLAlchemy with retry logic, connection health checks,
query helpers, and context managers.

Provides:
    - DBConnector class  — full-featured connection manager
    - execute_query()    — one-shot parameterised SELECT helper
    - execute_write()    — one-shot parameterised INSERT/UPDATE/DELETE helper
    - db_available()     — lightweight connectivity check
    - get_engine()       — shared engine singleton (same as config.py)

Design decisions:
    - Uses SQLAlchemy Core (text() queries) — no ORM, no model classes
    - All queries parameterised — zero raw string concatenation
    - Reads/writes use separate connection checkouts from the pool
    - Graceful degradation: all public methods return None/False on failure
      so callers can fall back to synthetic data without crashing

Usage:
    from utils.db_connector import DBConnector, execute_query

    # Option A — direct helper
    rows = execute_query("SELECT id, category FROM faults WHERE status = :s",
                         params={'s': 'OPEN'})

    # Option B — context manager
    with DBConnector() as db:
        df = db.read_dataframe("SELECT * FROM faults LIMIT 100")
        db.write("INSERT INTO ai_predictions (ds, yhat) VALUES (:ds, :y)",
                 [{'ds': '2026-01-01', 'y': 22}])
"""

import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Union

import pandas as pd
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.pool import QueuePool

from config import Config

logger = logging.getLogger('slt_ai.db')

# ─── Engine singleton ─────────────────────────────────────────────────────────
_engine = None
_engine_healthy = False


def _build_engine():
    """Create a new SQLAlchemy engine with the current Config."""
    return create_engine(
        Config.db_url(),
        poolclass=QueuePool,
        pool_size=Config.DB_POOL_SIZE,
        pool_recycle=Config.DB_POOL_RECYCLE,
        pool_pre_ping=True,
        pool_timeout=10,
        connect_args={
            'connect_timeout': 8,
            'charset': 'utf8mb4',
            'autocommit': False,
        },
        echo=False,
    )


def get_engine():
    """
    Return the shared SQLAlchemy engine (singleton).
    Creates the engine on first call; tests connectivity.
    Returns None if the database is unreachable.
    """
    global _engine, _engine_healthy

    if _engine is not None and _engine_healthy:
        return _engine

    try:
        _engine = _build_engine()
        with _engine.connect() as conn:
            conn.execute(text('SELECT 1'))
        _engine_healthy = True
        logger.info(
            f"✅ DB engine ready: "
            f"{Config.DB_HOST}:{Config.DB_PORT}/{Config.DB_NAME}"
        )
        return _engine
    except Exception as exc:
        logger.warning(f"⚠️  Database not reachable: {exc}")
        _engine = None
        _engine_healthy = False
        return None


def db_available() -> bool:
    """
    Fast connectivity check.
    Returns True only if a real SELECT 1 succeeds right now.
    """
    engine = get_engine()
    if engine is None:
        return False
    try:
        with engine.connect() as conn:
            conn.execute(text('SELECT 1'))
        return True
    except Exception:
        global _engine_healthy
        _engine_healthy = False
        return False


def reset_engine():
    """
    Dispose the current engine and force a fresh connection on next call.
    Useful after a DB restart.
    """
    global _engine, _engine_healthy
    if _engine is not None:
        try:
            _engine.dispose()
        except Exception:
            pass
    _engine = None
    _engine_healthy = False
    logger.info("Engine disposed — will reconnect on next use")


# ─────────────────────────────────────────────────────────────────────────────
# Standalone helper functions (thin wrappers — prefer DBConnector class)
# ─────────────────────────────────────────────────────────────────────────────

def execute_query(
    sql: str,
    params: Optional[Dict[str, Any]] = None,
    as_dataframe: bool = False
) -> Optional[Union[List[dict], pd.DataFrame]]:
    """
    Execute a parameterised SELECT query.

    Args:
        sql:          SQL string with :named placeholders.
        params:       Dict of parameter values.
        as_dataframe: If True, return a pandas DataFrame instead of list of dicts.

    Returns:
        List of row dicts, DataFrame, or None on error.

    Example:
        rows = execute_query(
            "SELECT id, latitude, longitude FROM faults "
            "WHERE created_at > :since AND latitude IS NOT NULL",
            params={'since': '2025-01-01'}
        )
    """
    engine = get_engine()
    if engine is None:
        return pd.DataFrame() if as_dataframe else None
    try:
        if as_dataframe:
            return pd.read_sql(text(sql), engine, params=params or {})
        with engine.connect() as conn:
            result = conn.execute(text(sql), params or {})
            return [dict(row._mapping) for row in result]
    except SQLAlchemyError as exc:
        logger.error(f"execute_query error: {exc}")
        return pd.DataFrame() if as_dataframe else None


def execute_write(
    sql: str,
    params: Optional[Union[Dict, List[Dict]]] = None
) -> bool:
    """
    Execute a parameterised INSERT / UPDATE / DELETE.

    Args:
        sql:    SQL string with :named placeholders.
        params: Single dict or list of dicts (for executemany).

    Returns:
        True on success, False on error.

    Example:
        ok = execute_write(
            "INSERT INTO ai_predictions (ds, yhat, model_version) "
            "VALUES (:ds, :yhat, :version)",
            [{'ds': '2026-01-01', 'yhat': 22.5, 'version': '1.0'}]
        )
    """
    engine = get_engine()
    if engine is None:
        return False
    try:
        with engine.begin() as conn:
            if isinstance(params, list):
                conn.execute(text(sql), params)
            else:
                conn.execute(text(sql), params or {})
        return True
    except SQLAlchemyError as exc:
        logger.error(f"execute_write error: {exc}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# DBConnector class — context manager + richer API
# ─────────────────────────────────────────────────────────────────────────────

class DBConnector:
    """
    Full-featured database connector with retry logic and connection management.

    Usage as context manager:
        with DBConnector() as db:
            df = db.read_dataframe("SELECT * FROM faults LIMIT 10")

    Usage standalone:
        db = DBConnector()
        if db.available():
            rows = db.read_rows("SELECT id FROM faults WHERE status = :s",
                                {'s': 'OPEN'})
        db.close()
    """

    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        self._engine       = None
        self._max_retries  = max_retries
        self._retry_delay  = retry_delay
        self._connected    = False
        self._connect()

    # ── Connection lifecycle ──────────────────────────────────────────────────

    def _connect(self) -> bool:
        """Try to establish a DB engine, with retries."""
        for attempt in range(1, self._max_retries + 1):
            engine = get_engine()
            if engine is not None:
                self._engine    = engine
                self._connected = True
                return True
            if attempt < self._max_retries:
                logger.debug(f"DB connect attempt {attempt} failed — retrying in {self._retry_delay}s")
                time.sleep(self._retry_delay)

        logger.warning(
            f"DB unavailable after {self._max_retries} attempts — "
            "AI module will use synthetic data"
        )
        self._connected = False
        return False

    def available(self) -> bool:
        """Check if the DB engine is live and responsive."""
        return db_available()

    def close(self) -> None:
        """Release resources (no-op for shared pool engine)."""
        self._engine    = None
        self._connected = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False   # do not suppress exceptions

    # ── Read operations ───────────────────────────────────────────────────────

    def read_dataframe(
        self,
        sql: str,
        params: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Execute a SELECT and return a pandas DataFrame.
        Returns empty DataFrame if connection is unavailable.

        Args:
            sql:    Parameterised SQL (use :name placeholders).
            params: Dict of parameter values.

        Returns:
            pd.DataFrame (may be empty, never None).
        """
        if not self._connected or self._engine is None:
            return pd.DataFrame()
        try:
            return pd.read_sql(text(sql), self._engine, params=params or {})
        except SQLAlchemyError as exc:
            logger.error(f"read_dataframe error: {exc}")
            return pd.DataFrame()

    def read_rows(
        self,
        sql: str,
        params: Optional[Dict[str, Any]] = None
    ) -> List[dict]:
        """
        Execute a SELECT and return a list of row dicts.
        Returns empty list if connection is unavailable.
        """
        if not self._connected or self._engine is None:
            return []
        try:
            with self._engine.connect() as conn:
                result = conn.execute(text(sql), params or {})
                return [dict(row._mapping) for row in result]
        except SQLAlchemyError as exc:
            logger.error(f"read_rows error: {exc}")
            return []

    def read_scalar(
        self,
        sql: str,
        params: Optional[Dict[str, Any]] = None,
        default: Any = None
    ) -> Any:
        """
        Execute a SELECT that returns a single value.

        Args:
            sql:     Query that returns one column in one row.
            params:  Parameter dict.
            default: Value to return if no rows or error.

        Returns:
            Single scalar value or default.
        """
        if not self._connected or self._engine is None:
            return default
        try:
            with self._engine.connect() as conn:
                result = conn.execute(text(sql), params or {})
                row = result.fetchone()
                return row[0] if row else default
        except SQLAlchemyError as exc:
            logger.error(f"read_scalar error: {exc}")
            return default

    # ── Write operations ──────────────────────────────────────────────────────

    def write(
        self,
        sql: str,
        params: Optional[Union[Dict, List[Dict]]] = None
    ) -> bool:
        """
        Execute an INSERT / UPDATE / DELETE within an auto-commit transaction.

        Args:
            sql:    Parameterised SQL.
            params: Single dict or list of dicts for batch execution.

        Returns:
            True on success, False on failure.
        """
        if not self._connected or self._engine is None:
            return False
        try:
            with self._engine.begin() as conn:
                if isinstance(params, list):
                    conn.execute(text(sql), params)
                else:
                    conn.execute(text(sql), params or {})
            return True
        except SQLAlchemyError as exc:
            logger.error(f"write error: {exc}")
            return False

    def bulk_insert_dataframe(
        self,
        df: pd.DataFrame,
        table_name: str,
        if_exists: str = 'append',
        chunksize: int = 500
    ) -> bool:
        """
        Insert a DataFrame into a MySQL table using pandas to_sql.

        Args:
            df:         DataFrame to insert.
            table_name: Target table name.
            if_exists:  'append' (default) | 'replace' | 'fail'.
            chunksize:  Rows per batch (default 500).

        Returns:
            True on success, False on failure.
        """
        if not self._connected or self._engine is None or df.empty:
            return False
        try:
            df.to_sql(
                table_name,
                self._engine,
                if_exists=if_exists,
                index=False,
                chunksize=chunksize,
                method='multi',
            )
            logger.info(f"Inserted {len(df)} rows into {table_name}")
            return True
        except SQLAlchemyError as exc:
            logger.error(f"bulk_insert_dataframe error on {table_name}: {exc}")
            return False

    # ── Introspection ─────────────────────────────────────────────────────────

    def table_exists(self, table_name: str) -> bool:
        """Check whether a table exists in the configured database."""
        if not self._connected or self._engine is None:
            return False
        try:
            insp = inspect(self._engine)
            return table_name in insp.get_table_names()
        except SQLAlchemyError:
            return False

    def row_count(self, table_name: str) -> int:
        """Return the approximate row count for a table. Returns 0 on error."""
        return self.read_scalar(
            f"SELECT COUNT(*) FROM `{table_name}`",
            default=0
        )

    def get_table_names(self) -> List[str]:
        """Return list of all table names in the database."""
        if not self._connected or self._engine is None:
            return []
        try:
            return inspect(self._engine).get_table_names()
        except SQLAlchemyError:
            return []

    # ── Health helpers ────────────────────────────────────────────────────────

    def ping(self) -> bool:
        """Execute SELECT 1 — True if responsive, False otherwise."""
        return self.read_scalar('SELECT 1', default=None) == 1

    def get_db_info(self) -> dict:
        """
        Return database server metadata.
        Useful for health check and admin dashboard.
        """
        if not self._connected or self._engine is None:
            return {'available': False}

        info = {'available': True, 'host': Config.DB_HOST, 'database': Config.DB_NAME}
        try:
            row = self.read_rows(
                "SELECT VERSION() AS version, NOW() AS server_time"
            )
            if row:
                info['version']     = row[0].get('version')
                info['server_time'] = str(row[0].get('server_time'))
        except Exception:
            pass

        # Table row counts
        key_tables = ['faults', 'jobs', 'users', 'technician_locations',
                      'payments', 'materials', 'kpi_scores']
        counts = {}
        for tbl in key_tables:
            if self.table_exists(tbl):
                counts[tbl] = self.row_count(tbl)
        info['table_counts'] = counts
        return info
