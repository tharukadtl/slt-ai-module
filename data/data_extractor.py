"""
data/data_extractor.py — MySQL Data Extraction
===============================================
Fetches raw data from the slt_fieldops_db MySQL database.
All queries are parameterised (no string concatenation) for security.

Tables read:
    faults, jobs, users, opmcs, technician_locations,
    kpi_scores, attendance, payments, materials, material_usage

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
                 created_at, opmc_id, circuit_id, nearest_exchange_id,
                 nearest_exchange_distance_km

        The last three (H1d, 2026-08-21) are additive, read-only columns —
        they let a clustered fault be cross-referenced against its stable,
        human-meaningful Exchange/Circuit (fieldops' real infrastructure
        hierarchy) alongside its numeric clusterId, which shifts on every
        K-Means refit. Confirmed via direct investigation before adding
        (see QA_Compliance_Consolidated_Report.md's H1d entry): neither
        KMeansClusterer.get_cluster_features() (explicit [latitude,
        longitude] column slice), DataCleaner.clean_gps_points() (its
        dropna/drop_duplicates are both scoped to the lat/lng columns by
        name, never a blanket subset-less call), nor
        _build_cluster_summaries() (touches only cluster_id/category/
        latitude/longitude) reference these three columns at all — adding
        them here cannot change what rows survive cleaning or what feeds
        the actual K-Means fit. Most faults will carry NULL circuit_id
        (nothing sets it yet outside the new manual-attach endpoint) and
        possibly NULL nearest_exchange_id (only set when GPS was present
        AND a geocoded Exchange existed at fault-creation time) — both
        expected and harmless, never filtered on here.

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
                f.opmc_id,
                f.circuit_id,
                f.nearest_exchange_id,
                f.nearest_exchange_distance_km
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

        Columns: technician_id, full_name, phone, opmc_id,
                 latitude, longitude, status, last_seen, current_job_id
        """
        if not self._db_ok:
            return None

        # MAX(...) on the non-grouped columns keeps this valid under MySQL's
        # default ONLY_FULL_GROUP_BY mode. A technician can have more than one
        # technician_locations row (history), and the updated_at >= 2-hour
        # filter typically leaves at most one recent row per technician
        # anyway, so MAX() effectively just unwraps that single row's values.
        sql = text("""
            SELECT
                u.id                        AS technician_id,
                u.full_name,
                u.phone,
                u.opmc_id,
                MAX(tl.latitude)            AS latitude,
                MAX(tl.longitude)           AS longitude,
                MAX(tl.technician_status)   AS status,
                MAX(tl.updated_at)          AS last_seen,
                MAX(j.id)                   AS current_job_id
            FROM users u
            INNER JOIN technician_locations tl
                ON tl.user_id = u.id
            LEFT JOIN jobs j
                ON j.technician_id = u.id
                AND j.status IN ('ACCEPTED','TRAVELLING','IN_PROGRESS')
            WHERE
                u.role IN ('TECHNICIAN','TEAM_LEAD')
                AND u.is_active = 1
                AND tl.updated_at >= DATE_SUB(NOW(), INTERVAL 2 HOUR)
                AND tl.latitude  BETWEEN :lat_min AND :lat_max
                AND tl.longitude BETWEEN :lng_min AND :lng_max
            GROUP BY u.id, u.full_name, u.phone, u.opmc_id
            ORDER BY last_seen DESC
        """)
        try:
            df = pd.read_sql(sql, self.engine, params={
                'lat_min': Config.SL_LAT_MIN, 'lat_max': Config.SL_LAT_MAX,
                'lng_min': Config.SL_LNG_MIN, 'lng_max': Config.SL_LNG_MAX,
            })
            df['latitude']  = pd.to_numeric(df['latitude'],  errors='coerce')
            df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
            df = df.dropna(subset=['latitude', 'longitude'])
            # current_job_id comes from a LEFT JOIN — technicians with no
            # active job get NULL, which pandas widens to float NaN (since
            # the column also holds real integer ids). A bare NaN is not
            # valid JSON and breaks JSON.parse() on the frontend, so make
            # the "no job" case an explicit None instead of NaN.
            if 'current_job_id' in df.columns:
                df['current_job_id'] = df['current_job_id'].astype(object).where(
                    df['current_job_id'].notna(), None
                )
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
                u.opmc_id,
                o.name          AS opmc_name,
                u.is_active
            FROM users u
            LEFT JOIN opmcs o ON o.id = u.opmc_id
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
                j.technician_id                              AS technician_id,
                DATE(j.created_at)                           AS ds,
                COUNT(*)                                     AS jobs_assigned,
                SUM(CASE WHEN j.status='COMPLETED' THEN 1 ELSE 0 END)
                                                             AS jobs_completed,
                AVG(TIMESTAMPDIFF(MINUTE, j.created_at, j.completed_at))
                                                             AS avg_duration_min
            FROM jobs j
            WHERE
                j.created_at >= DATE_SUB(CURDATE(), INTERVAL :days DAY)
                AND j.technician_id IS NOT NULL
            GROUP BY j.technician_id, DATE(j.created_at)
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
    # FR-33 (SRS 5.6.8) — PREDICTIVE RESOURCE PLANNING INPUTS
    # ─────────────────────────────────────────────────────────────────────────

    def get_technician_shift_performance(self, days_back: int = 180) -> Optional[pd.DataFrame]:
        """
        Completed-job counts per technician per shift window (FR-33 Technician
        Suggestion: predicted fault count ÷ avg jobs completed per Technician
        per shift). Buckets jobs.completed_at's hour into the three SRS 5.6.8
        shift windows — Morning 06:00-12:00, Afternoon 12:00-18:00,
        Evening 18:00-22:00; completions outside those hours (22:00-06:00)
        are deliberately excluded, not folded into a nearest window, since
        the SRS only defines these three.

        Columns: technician_id, shift, jobs_completed, days_covered
            jobs_completed: total completions in that technician/shift bucket
                            over the window.
            days_covered:   distinct calendar days with >=1 completion in
                            that bucket — the caller (FR-33 combination
                            logic) decides how to turn this into an average
                            (e.g. jobs_completed / days_covered for "per day
                            they worked that shift", or against the full
                            days_back window for "per calendar day including
                            zero-job days") — this method only supplies the
                            raw counts, not the averaging policy.
        """
        if not self._db_ok:
            return None

        sql = text("""
            SELECT
                technician_id,
                shift,
                COUNT(*)                       AS jobs_completed,
                COUNT(DISTINCT completed_date) AS days_covered
            FROM (
                SELECT
                    technician_id,
                    DATE(completed_at) AS completed_date,
                    CASE
                        WHEN HOUR(completed_at) >= 6  AND HOUR(completed_at) < 12 THEN 'MORNING'
                        WHEN HOUR(completed_at) >= 12 AND HOUR(completed_at) < 18 THEN 'AFTERNOON'
                        WHEN HOUR(completed_at) >= 18 AND HOUR(completed_at) < 22 THEN 'EVENING'
                        ELSE NULL
                    END AS shift
                FROM jobs
                WHERE status = 'COMPLETED'
                  AND completed_at IS NOT NULL
                  AND completed_at >= DATE_SUB(NOW(), INTERVAL :days DAY)
                  AND technician_id IS NOT NULL
            ) shifted
            WHERE shift IS NOT NULL
            GROUP BY technician_id, shift
            ORDER BY technician_id, shift
        """)
        try:
            df = pd.read_sql(sql, self.engine, params={'days': days_back})
            logger.info(f"Extracted shift performance: {len(df)} technician-shift rows")
            return df
        except Exception as e:
            logger.error(f"get_technician_shift_performance error: {e}")
            return None

    def get_material_usage_with_location(self, days_back: int = 365) -> Optional[pd.DataFrame]:
        """
        Historical material usage joined to each job's fault category and
        GPS location (FR-33 Material Suggestion: current stock, 5.5.3,
        cross-checked against historical average usage per fault category
        per zone, from MaterialUsage records, Annexure B).

        Join path: material_usage -> jobs (job_id) -> faults (fault_id), for
        category. Location is read from faults.latitude/longitude, NOT
        jobs.latitude/longitude — verified against live data that the jobs
        table's own lat/lng columns are NULL in practice even though the
        column exists, while faults.latitude/longitude are always populated.

        This method does the JOIN only — it does not assign a K-Means zone
        to each row. That requires the already-fitted KMeansClusterer's
        model (a modelling concern, not a DB-extraction one) to predict a
        cluster for each (latitude, longitude) pair; that cross-referencing
        is FR-33's resource-plan combination logic (Stage 2), not this
        method's job.

        Columns: material_id, material_name, fault_category, latitude,
                 longitude, quantity_used, usage_date
        """
        if not self._db_ok:
            return None

        sql = text("""
            SELECT
                mu.material_id,
                mat.name    AS material_name,
                f.category  AS fault_category,
                f.latitude,
                f.longitude,
                mu.quantity_used,
                mu.created_at AS usage_date
            FROM material_usage mu
            INNER JOIN jobs   j ON mu.job_id     = j.id
            INNER JOIN faults f ON j.fault_id    = f.id
            LEFT  JOIN materials mat ON mu.material_id = mat.id
            WHERE mu.created_at >= DATE_SUB(NOW(), INTERVAL :days DAY)
              AND f.latitude  IS NOT NULL
              AND f.longitude IS NOT NULL
            ORDER BY mu.created_at
        """)
        try:
            df = pd.read_sql(sql, self.engine, params={'days': days_back})
            df['quantity_used'] = pd.to_numeric(df['quantity_used'], errors='coerce')
            logger.info(f"Extracted material usage with location: {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"get_material_usage_with_location error: {e}")
            return None

    def get_fault_shift_distribution(self, days_back: int = 180) -> Optional[pd.DataFrame]:
        """
        Historical share of fault OCCURRENCES per shift window (FR-33 Time
        Window Prediction: distributes Prophet's daily forecast into
        Morning/Afternoon/Evening windows using each window's real
        historical share of faults.created_at — deliberately NOT
        get_technician_shift_performance's jobs.completed_at, which is a
        different signal (when a technician finished work, not when the
        fault was reported) — the SRS's daily forecast is itself trained on
        faults.created_at (see get_fault_time_series), so the shift split
        must use the same timestamp basis to be consistent.

        Same three shift windows as get_technician_shift_performance —
        Morning 06:00-12:00, Afternoon 12:00-18:00, Evening 18:00-22:00;
        faults reported outside those hours (22:00-06:00) are excluded
        from both the numerator and denominator, so the returned
        fault_count values are shares of "faults reported within a defined
        shift window", not shares of all faults — callers should normalise
        by the sum of THIS DataFrame's fault_count, not by a separately
        fetched all-hours total.

        Columns: shift, fault_count
        """
        if not self._db_ok:
            return None

        sql = text("""
            SELECT shift, COUNT(*) AS fault_count
            FROM (
                SELECT
                    CASE
                        WHEN HOUR(created_at) >= 6  AND HOUR(created_at) < 12 THEN 'MORNING'
                        WHEN HOUR(created_at) >= 12 AND HOUR(created_at) < 18 THEN 'AFTERNOON'
                        WHEN HOUR(created_at) >= 18 AND HOUR(created_at) < 22 THEN 'EVENING'
                        ELSE NULL
                    END AS shift
                FROM faults
                WHERE created_at IS NOT NULL
                  AND created_at >= DATE_SUB(NOW(), INTERVAL :days DAY)
            ) shifted
            WHERE shift IS NOT NULL
            GROUP BY shift
            ORDER BY shift
        """)
        try:
            df = pd.read_sql(sql, self.engine, params={'days': days_back})
            logger.info(f"Extracted fault shift distribution: {len(df)} shift rows")
            return df
        except Exception as e:
            logger.error(f"get_fault_shift_distribution error: {e}")
            return None

    def get_material_stock(self) -> Optional[pd.DataFrame]:
        """
        Current stock level per active material (FR-33 Material Suggestion,
        SRS 5.5.3) — used to flag hotspots/materials where predicted demand
        would exceed what's on hand.

        Columns: material_id, material_name, current_stock, unit
        """
        if not self._db_ok:
            return None

        sql = text("""
            SELECT
                id   AS material_id,
                name AS material_name,
                current_stock,
                unit
            FROM materials
            WHERE is_active = 1
        """)
        try:
            df = pd.read_sql(sql, self.engine)
            df['current_stock'] = pd.to_numeric(df['current_stock'], errors='coerce')
            logger.info(f"Extracted material stock: {len(df)} materials")
            return df
        except Exception as e:
            logger.error(f"get_material_stock error: {e}")
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # METADATA
    # ─────────────────────────────────────────────────────────────────────────

    def get_branches(self) -> Optional[pd.DataFrame]:
        """
        Returns all OPMCs with name, province, GPS if stored.

        Kept the name get_branches() (not renamed to get_opmcs()) since this
        method has zero callers anywhere in the module (confirmed by search)
        — renaming an unreferenced method name isn't needed to complete the
        table/column rename this method's SQL required.

        Note (pre-existing, unrelated to the OPMC rename — NOT fixed here):
        this query selects a `region` column, but the branches/opmcs table
        has never had one — CreateBranchRequest/CreateOpmcRequest's `region`
        field is a request-only fallback that gets folded into the `province`
        column server-side (see OpmcService.mapRequestToEntity), never
        persisted under its own name. This SELECT would already have failed
        with "Unknown column 'region'" before this rename, on the old
        `branches` table just the same. Left as found since this method is
        unreferenced and fixing unrelated bugs isn't this task's scope.
        """
        if not self._db_ok:
            return None
        sql = text("SELECT id, name, region, code FROM opmcs ORDER BY name")
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
