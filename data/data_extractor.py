"""
data/data_extractor.py — MySQL Data Extraction
===============================================
Fetches raw data from the slt_fieldops_db MySQL database.
All queries are parameterised (no string concatenation) for security.

Tables read:
    faults, jobs, users, branches, technician_locations,
    kpi_scores, attendance, payments, material_usages

Usage:
    from data.data_extractor import DataExtractor
    extractor = DataExtractor()
    df = extractor.get_fault_time_series(days_back=365)
"""

import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
from sqlalchemy import text

from config import get_db_engine, Config

logger = logging.getLogger('slt_ai.extractor')


class DataExtractor:
    """
    Reads production data from MySQL.
    Falls back gracefully when DB is not available.
    """

    def __init__(self):
        self.engine = get_db_engine()
        self._db_ok = self.engine is not None

    # ─────────────────────────────────────────────────────────────────────────
    # FAULT DATA
    # ─────────────────────────────────────────────────────────────────────────

    def get_fault_time_series(self, days_back: int = 365) -> Optional[pd.DataFrame]:
        """
        Returns daily fault counts for Prophet training.

        Columns: ds (date), y (fault_count)

        Args:
            days_back: How many days of history to retrieve (default 365).

        Returns:
            DataFrame with columns [ds, y] or None if DB unavailable.
        """
        if not self._db_ok:
            return None

        sql = text("""
            SELECT
                DATE(created_at)             AS ds,
                COUNT(*)                     AS y
            FROM faults
            WHERE
                created_at >= DATE_SUB(CURDATE(), INTERVAL :days DAY)
                AND created_at IS NOT NULL
            GROUP BY DATE(created_at)
            ORDER BY ds
        """)
        try:
            df = pd.read_sql(sql, self.engine, params={'days': days_back})
            df['ds'] = pd.to_datetime(df['ds'])
            df['y']  = df['y'].astype(float)
            logger.info(f"Extracted fault time-series: {len(df)} days")
            return df
        except Exception as e:
            logger.error(f"get_fault_time_series error: {e}")
            return None

    def get_faults_with_location(
        self,
        days_back: int = 180,
        status_filter: Optional[list] = None
    ) -> Optional[pd.DataFrame]:
        """
        Returns faults with GPS coordinates for K-Means clustering.

        Columns: id, latitude, longitude, category, status, priority,
                 created_at, branch_id

        Args:
            days_back:      Days of history (default 180).
            status_filter:  List of statuses to include. Default: all.

        Returns:
            DataFrame or None.
        """
        if not self._db_ok:
            return None

        status_clause = ""
        params: dict = {'days': days_back}
        if status_filter:
            placeholders = ', '.join(f':s{i}' for i in range(len(status_filter)))
            status_clause = f"AND f.status IN ({placeholders})"
            for i, s in enumerate(status_filter):
                params[f's{i}'] = s

        sql = text(f"""
            SELECT
                f.id,
                f.latitude,
                f.longitude,
                f.category,
                f.status,
                f.priority,
                f.created_at,
                f.branch_id
            FROM faults f
            WHERE
                f.created_at >= DATE_SUB(CURDATE(), INTERVAL :days DAY)
                AND f.latitude  IS NOT NULL
                AND f.longitude IS NOT NULL
                AND f.latitude  BETWEEN :lat_min AND :lat_max
                AND f.longitude BETWEEN :lng_min AND :lng_max
                {status_clause}
            ORDER BY f.created_at DESC
        """)
        params.update({
            'lat_min': Config.SL_LAT_MIN, 'lat_max': Config.SL_LAT_MAX,
            'lng_min': Config.SL_LNG_MIN, 'lng_max': Config.SL_LNG_MAX,
        })
        try:
            df = pd.read_sql(sql, self.engine, params=params)
            df['latitude']  = pd.to_numeric(df['latitude'],  errors='coerce')
            df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
            df = df.dropna(subset=['latitude', 'longitude'])
            logger.info(f"Extracted {len(df)} faults with GPS")
            return df
        except Exception as e:
            logger.error(f"get_faults_with_location error: {e}")
            return None

    def get_fault_category_breakdown(self, days_back: int = 90) -> Optional[pd.DataFrame]:
        """
        Returns fault counts grouped by category and date.
        Useful for category-level forecasting.
        """
        if not self._db_ok:
            return None

        sql = text("""
            SELECT
                DATE(created_at) AS ds,
                category,
                COUNT(*)         AS count
            FROM faults
            WHERE created_at >= DATE_SUB(CURDATE(), INTERVAL :days DAY)
            GROUP BY DATE(created_at), category
            ORDER BY ds, category
        """)
        try:
            df = pd.read_sql(sql, self.engine, params={'days': days_back})
            df['ds'] = pd.to_datetime(df['ds'])
            logger.info(f"Extracted category breakdown: {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"get_fault_category_breakdown error: {e}")
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # TECHNICIAN LOCATION DATA (for route optimisation)
    # ─────────────────────────────────────────────────────────────────────────

    def get_available_technicians(self) -> Optional[pd.DataFrame]:
        """
        Returns technicians with their current GPS positions.
        Used by Dijkstra route optimiser.

        Columns: technician_id, full_name, phone, branch_id,
                 latitude, longitude, status, last_seen, current_job_id
        """
        if not self._db_ok:
            return None

        sql = text("""
            SELECT
                u.id                        AS technician_id,
                u.full_name,
                u.phone,
                u.branch_id,
                tl.latitude,
                tl.longitude,
                tl.status,
                tl.updated_at               AS last_seen,
                j.id                        AS current_job_id
            FROM users u
            INNER JOIN technician_locations tl
                ON tl.technician_id = u.id
            LEFT JOIN jobs j
                ON j.assigned_to_id = u.id
                AND j.status IN ('ACCEPTED','TRAVELLING','IN_PROGRESS')
            WHERE
                u.role IN ('TECHNICIAN','TEAM_LEAD')
                AND u.is_active = 1
                AND tl.updated_at >= DATE_SUB(NOW(), INTERVAL 2 HOUR)
                AND tl.latitude  BETWEEN :lat_min AND :lat_max
                AND tl.longitude BETWEEN :lng_min AND :lng_max
            GROUP BY u.id
            ORDER BY tl.updated_at DESC
        """)
        try:
            df = pd.read_sql(sql, self.engine, params={
                'lat_min': Config.SL_LAT_MIN, 'lat_max': Config.SL_LAT_MAX,
                'lng_min': Config.SL_LNG_MIN, 'lng_max': Config.SL_LNG_MAX,
            })
            df['latitude']  = pd.to_numeric(df['latitude'],  errors='coerce')
            df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
            df = df.dropna(subset=['latitude', 'longitude'])
            logger.info(f"Found {len(df)} technicians with recent GPS")
            return df
        except Exception as e:
            logger.error(f"get_available_technicians error: {e}")
            return None

    def get_all_technicians(self) -> Optional[pd.DataFrame]:
        """
        Returns all active technicians regardless of GPS availability.
        Fallback for route optimiser when live locations are missing.
        """
        if not self._db_ok:
            return None

        sql = text("""
            SELECT
                u.id            AS technician_id,
                u.full_name,
                u.phone,
                u.branch_id,
                b.name          AS branch_name,
                u.is_active
            FROM users u
            LEFT JOIN branches b ON b.id = u.branch_id
            WHERE
                u.role IN ('TECHNICIAN','TEAM_LEAD')
                AND u.is_active = 1
            ORDER BY u.full_name
        """)
        try:
            df = pd.read_sql(sql, self.engine)
            logger.info(f"Loaded {len(df)} technicians")
            return df
        except Exception as e:
            logger.error(f"get_all_technicians error: {e}")
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # KPI & PERFORMANCE DATA
    # ─────────────────────────────────────────────────────────────────────────

    def get_technician_performance(self, days_back: int = 90) -> Optional[pd.DataFrame]:
        """
        Returns per-technician completion and satisfaction data.
        Used as regressors to improve Prophet forecast accuracy.
        """
        if not self._db_ok:
            return None

        sql = text("""
            SELECT
                j.assigned_to_id                             AS technician_id,
                DATE(j.created_at)                           AS ds,
                COUNT(*)                                     AS jobs_assigned,
                SUM(CASE WHEN j.status='COMPLETED' THEN 1 ELSE 0 END)
                                                             AS jobs_completed,
                AVG(TIMESTAMPDIFF(MINUTE, j.created_at, j.completed_at))
                                                             AS avg_duration_min
            FROM jobs j
            WHERE
                j.created_at >= DATE_SUB(CURDATE(), INTERVAL :days DAY)
                AND j.assigned_to_id IS NOT NULL
            GROUP BY j.assigned_to_id, DATE(j.created_at)
            ORDER BY ds
        """)
        try:
            df = pd.read_sql(sql, self.engine, params={'days': days_back})
            df['ds'] = pd.to_datetime(df['ds'])
            logger.info(f"Extracted performance data: {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"get_technician_performance error: {e}")
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # METADATA
    # ─────────────────────────────────────────────────────────────────────────

    def get_branches(self) -> Optional[pd.DataFrame]:
        """Returns all branches with name, region, GPS if stored."""
        if not self._db_ok:
            return None
        sql = text("SELECT id, name, region, code FROM branches ORDER BY name")
        try:
            return pd.read_sql(sql, self.engine)
        except Exception as e:
            logger.error(f"get_branches error: {e}")
            return None

    def count_records(self) -> dict:
        """
        Returns record counts from key tables.
        Used by /api/ai/health to confirm data availability.
        """
        if not self._db_ok:
            return {'db_available': False}

        counts = {'db_available': True}
        tables = {
            'faults': 'SELECT COUNT(*) FROM faults',
            'faults_with_gps': (
                'SELECT COUNT(*) FROM faults '
                'WHERE latitude IS NOT NULL AND longitude IS NOT NULL'
            ),
            'technicians': (
                "SELECT COUNT(*) FROM users "
                "WHERE role IN ('TECHNICIAN','TEAM_LEAD') AND is_active=1"
            ),
            'recent_locations': (
                'SELECT COUNT(*) FROM technician_locations '
                'WHERE updated_at >= DATE_SUB(NOW(), INTERVAL 2 HOUR)'
            ),
        }
        with self.engine.connect() as conn:
            for key, query in tables.items():
                try:
                    result = conn.execute(text(query))
                    counts[key] = result.scalar()
                except Exception:
                    counts[key] = 0
        return counts
