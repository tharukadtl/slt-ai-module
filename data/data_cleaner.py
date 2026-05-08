"""
data/data_cleaner.py — Data Cleaning & Preparation
====================================================
Cleans raw DataFrames from data_extractor.py before feeding
to Prophet, K-Means, or the Dijkstra router.

Operations:
  - Remove nulls and duplicates
  - Fix data types
  - Clip geographic outliers to Sri Lanka bounds
  - Fill missing dates in time-series (zero-fill)
  - Remove statistical outliers (IQR method)
  - Validate data sufficiency for model training

Usage:
    from data.data_cleaner import DataCleaner
    cleaner = DataCleaner()
    clean_df = cleaner.clean_time_series(raw_df)
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Tuple

from config import Config

logger = logging.getLogger('slt_ai.cleaner')


class DataCleaner:
    """Stateless cleaner — all methods are pure functions on DataFrames."""

    # ─────────────────────────────────────────────────────────────────────────
    # TIME-SERIES CLEANING (Prophet input)
    # ─────────────────────────────────────────────────────────────────────────

    def clean_time_series(
        self,
        df: pd.DataFrame,
        date_col: str = 'ds',
        value_col: str = 'y',
        fill_zeros: bool = True,
        remove_outliers: bool = True,
        min_rows: int = 30
    ) -> Tuple[Optional[pd.DataFrame], dict]:
        """
        Clean a time-series DataFrame for Prophet.

        Args:
            df:               Raw DataFrame with date + count columns.
            date_col:         Name of the date column (default 'ds').
            value_col:        Name of the value column (default 'y').
            fill_zeros:       Fill missing dates with y=0 (default True).
            remove_outliers:  Remove statistical outliers using IQR.
            min_rows:         Minimum rows required after cleaning.

        Returns:
            Tuple of (cleaned_df | None, stats_dict)
        """
        stats = {
            'input_rows': len(df) if df is not None else 0,
            'issues': [],
        }

        if df is None or df.empty:
            stats['issues'].append('Input DataFrame is None or empty')
            return None, stats

        df = df.copy()

        # 1. Rename to standard columns
        rename_map = {}
        if date_col != 'ds':
            rename_map[date_col] = 'ds'
        if value_col != 'y':
            rename_map[value_col] = 'y'
        if rename_map:
            df = df.rename(columns=rename_map)

        # 2. Parse dates
        df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
        df['y']  = pd.to_numeric(df['y'], errors='coerce')

        # 3. Drop rows with null ds or y
        before = len(df)
        df = df.dropna(subset=['ds', 'y'])
        dropped = before - len(df)
        if dropped > 0:
            stats['issues'].append(f"Dropped {dropped} rows with null ds/y")
            logger.debug(f"Dropped {dropped} null rows")

        # 4. Drop duplicates on date (keep last)
        before = len(df)
        df = df.drop_duplicates(subset=['ds'], keep='last')
        dupes = before - len(df)
        if dupes > 0:
            stats['issues'].append(f"Removed {dupes} duplicate dates")

        # 5. Clip negative values (count cannot be negative)
        neg_mask = df['y'] < 0
        if neg_mask.any():
            df.loc[neg_mask, 'y'] = 0
            stats['issues'].append(f"Clipped {neg_mask.sum()} negative values to 0")

        # 6. Remove statistical outliers using IQR method
        if remove_outliers and len(df) > 20:
            df, n_removed = self._remove_outliers_iqr(df, 'y', factor=3.0)
            if n_removed > 0:
                stats['issues'].append(f"Removed {n_removed} outlier rows")

        # 7. Sort by date
        df = df.sort_values('ds').reset_index(drop=True)

        # 8. Fill missing dates with zero
        if fill_zeros and len(df) > 1:
            df = self._fill_missing_dates(df)
            logger.debug(f"After zero-fill: {len(df)} rows")

        # 9. Select only required columns
        df = df[['ds', 'y']].copy()

        stats['output_rows']  = len(df)
        stats['date_min']     = str(df['ds'].min().date()) if len(df) > 0 else None
        stats['date_max']     = str(df['ds'].max().date()) if len(df) > 0 else None
        stats['mean_y']       = round(float(df['y'].mean()), 2) if len(df) > 0 else 0
        stats['total_faults'] = int(df['y'].sum()) if len(df) > 0 else 0
        stats['sufficient']   = len(df) >= min_rows

        if not stats['sufficient']:
            msg = (f"Insufficient data: {len(df)} rows < {min_rows} required. "
                   "Will use synthetic data.")
            stats['issues'].append(msg)
            logger.warning(msg)
            return None, stats

        logger.info(
            f"Time-series clean: {stats['input_rows']} → {stats['output_rows']} rows, "
            f"range {stats['date_min']} to {stats['date_max']}"
        )
        return df, stats

    # ─────────────────────────────────────────────────────────────────────────
    # GPS / LOCATION CLEANING (K-Means + Dijkstra input)
    # ─────────────────────────────────────────────────────────────────────────

    def clean_gps_points(
        self,
        df: pd.DataFrame,
        lat_col: str = 'latitude',
        lng_col: str = 'longitude',
        min_points: int = 20
    ) -> Tuple[Optional[pd.DataFrame], dict]:
        """
        Clean GPS coordinate DataFrame for K-Means clustering.

        Args:
            df:         Raw DataFrame with lat/lng columns.
            lat_col:    Latitude column name.
            lng_col:    Longitude column name.
            min_points: Minimum valid points required.

        Returns:
            Tuple of (cleaned_df | None, stats_dict)
        """
        stats = {'input_rows': len(df) if df is not None else 0, 'issues': []}

        if df is None or df.empty:
            stats['issues'].append('Input DataFrame is None or empty')
            return None, stats

        df = df.copy()

        # 1. Coerce to float
        df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
        df[lng_col] = pd.to_numeric(df[lng_col], errors='coerce')

        # 2. Drop nulls
        before = len(df)
        df = df.dropna(subset=[lat_col, lng_col])
        if before - len(df) > 0:
            stats['issues'].append(f"Dropped {before - len(df)} null GPS rows")

        # 3. Clip to Sri Lanka bounds
        out_of_bounds = (
            (df[lat_col] < Config.SL_LAT_MIN) | (df[lat_col] > Config.SL_LAT_MAX) |
            (df[lng_col] < Config.SL_LNG_MIN) | (df[lng_col] > Config.SL_LNG_MAX)
        )
        if out_of_bounds.any():
            df = df[~out_of_bounds]
            stats['issues'].append(
                f"Removed {out_of_bounds.sum()} out-of-SL-bounds GPS points"
            )

        # 4. Remove exact duplicates
        before = len(df)
        df = df.drop_duplicates(subset=[lat_col, lng_col])
        if before - len(df) > 0:
            stats['issues'].append(f"Removed {before - len(df)} duplicate GPS points")

        # 5. Rename to standard names
        if lat_col != 'latitude':
            df = df.rename(columns={lat_col: 'latitude'})
        if lng_col != 'longitude':
            df = df.rename(columns={lng_col: 'longitude'})

        stats['output_rows'] = len(df)
        stats['sufficient']  = len(df) >= min_points

        if not stats['sufficient']:
            msg = f"Insufficient GPS points: {len(df)} < {min_points}. Falling back to synthetic."
            stats['issues'].append(msg)
            logger.warning(msg)
            return None, stats

        logger.info(f"GPS clean: {stats['input_rows']} → {stats['output_rows']} points")
        return df, stats

    def clean_technician_locations(
        self,
        df: pd.DataFrame
    ) -> Optional[pd.DataFrame]:
        """
        Clean technician location records for route optimisation.
        Returns only records with valid GPS inside Sri Lanka.
        """
        if df is None or df.empty:
            return None

        df = df.copy()
        df['latitude']  = pd.to_numeric(df['latitude'],  errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')

        # Keep only valid, in-bounds positions
        valid = (
            df['latitude'].notna() &
            df['longitude'].notna() &
            df['latitude'].between(Config.SL_LAT_MIN, Config.SL_LAT_MAX) &
            df['longitude'].between(Config.SL_LNG_MIN, Config.SL_LNG_MAX)
        )
        df = df[valid].copy()
        logger.debug(f"Technician locations after clean: {len(df)}")
        return df if not df.empty else None

    # ─────────────────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _remove_outliers_iqr(
        self,
        df: pd.DataFrame,
        col: str,
        factor: float = 3.0
    ) -> Tuple[pd.DataFrame, int]:
        """
        Remove rows where `col` is outside [Q1 - factor*IQR, Q3 + factor*IQR].
        Uses factor=3.0 (extreme outliers only) to preserve genuine spikes.
        """
        Q1  = df[col].quantile(0.25)
        Q3  = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        mask  = (df[col] >= lower) & (df[col] <= upper)
        n_removed = (~mask).sum()
        return df[mask].copy(), n_removed

    def _fill_missing_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a complete date range and zero-fill any missing dates.
        Assumes df is sorted and has columns [ds, y].
        """
        full_range = pd.date_range(
            start=df['ds'].min(),
            end=df['ds'].max(),
            freq='D'
        )
        df = df.set_index('ds').reindex(full_range, fill_value=0)
        df.index.name = 'ds'
        df = df.reset_index()
        return df

    def validate_for_prophet(self, df: pd.DataFrame) -> dict:
        """
        Run quick validation checks on a cleaned df before passing to Prophet.

        Returns dict with keys: valid (bool), warnings (list), errors (list)
        """
        result = {'valid': True, 'warnings': [], 'errors': []}

        if df is None or df.empty:
            result['valid'] = False
            result['errors'].append('DataFrame is None or empty')
            return result

        # Check required columns
        for col in ['ds', 'y']:
            if col not in df.columns:
                result['valid'] = False
                result['errors'].append(f"Missing required column: {col}")

        if not result['valid']:
            return result

        n_rows = len(df)
        if n_rows < Config.FORECAST_MIN_HISTORY_DAYS:
            result['warnings'].append(
                f"Only {n_rows} days of data; {Config.FORECAST_MIN_HISTORY_DAYS}+ recommended for best accuracy"
            )

        n_zeros = (df['y'] == 0).sum()
        zero_pct = n_zeros / n_rows * 100
        if zero_pct > 50:
            result['warnings'].append(
                f"{zero_pct:.0f}% of days have zero faults — model may underfit"
            )

        date_gaps = df['ds'].diff().dt.days.dropna()
        large_gaps = (date_gaps > 7).sum()
        if large_gaps > 0:
            result['warnings'].append(
                f"{large_gaps} date gaps > 7 days found in time-series"
            )

        result['n_rows']     = n_rows
        result['date_range'] = f"{df['ds'].min().date()} to {df['ds'].max().date()}"
        result['mean_y']     = round(float(df['y'].mean()), 2)
        result['max_y']      = int(df['y'].max())

        return result
