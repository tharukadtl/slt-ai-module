"""
data/upload_manager.py — CSV Training Data Upload Validation
===============================================================
Validates and parses an admin-uploaded CSV of historical fault records
for the SRS 5.6.7 training pipeline (POST /api/ai/train). Enforces the
Stage A checks — row-count sanity cap, required columns, parseable
dates, the 24-month (2-year) training window, GPS bounds — and hands
back the validated DataFrame directly; it does not persist anything to
disk itself (training operates on the returned DataFrame, and each
upload's data is scoped to the single request that validated it — see
save_upload()'s docstring for why that matters).

Expected CSV columns (case-insensitive, a couple of aliases accepted):
    date          (aliases: ds, created_at, timestamp)   — required
    latitude      (alias: lat)                            — optional
    longitude     (aliases: lng, lon)                      — optional
    exchange_area (alias: exchangearea)                    — optional
    opmc_code     (alias: opmccode)                         — optional
    category                                                — optional

Only `date` is strictly required (forecasting only needs the date).
Clustering needs EITHER latitude/longitude (real K-Means geographic
clustering) OR exchange_area (categorical grouping — see
to_exchange_area_groups() and KMeansClusterer.cluster_by_exchange_area()).
Both are optional and independent of each other — a CSV with real GPS
columns is validated and shaped exactly as before this was added; GPS
takes priority when both are present (see app.py's _run_training_job).

exchange_area exists because SLT's real WFMS fault export (confirmed
against a real sample, not assumed) has NO latitude/longitude at all —
only short exchange-area codes (e.g. DGD, AD, KY). Exchange has no
coordinates in this system either (Stage C scaffolding only), so there
is no coordinate to geocode a code into; the codes are grouped directly
instead of being coerced through K-Means. opmc_code rides along as
passthrough metadata only — checked against this database's real
Opmc.code values (ABC-01/TES-10/NEG-18) and confirmed NOT to match the
real export's codes (KTOP/ADOP/KYOP/...), so no FK/join is attempted
here; it is carried through in case a future OPMC-scoped feature wants
the raw value, not resolved against a real Opmc row today.

Usage:
    from data.upload_manager import UploadManager
    um = UploadManager()
    summary, df = um.save_upload(request.files['file'])
    ts_df   = um.to_time_series(df)
    gps_df  = um.to_gps_points(df)
    area_df = um.to_exchange_area_groups(df)
"""

import logging
from typing import Optional

import pandas as pd

from config import Config

logger = logging.getLogger('slt_ai.upload')

# Column-name aliases → canonical name
_COLUMN_ALIASES = {
    'ds': 'date', 'created_at': 'date', 'timestamp': 'date',
    'lat': 'latitude',
    'lng': 'longitude', 'lon': 'longitude',
    'exchangearea': 'exchange_area',
    'opmccode': 'opmc_code',
}
_REQUIRED_COLUMNS = ['date']
_MAX_ROWS = 200_000  # sanity cap — a training CSV shouldn't need more than this
_MAX_MONTHS_BACK = 24  # SRS 5.6.7 — training data capped at 24 months (2 years) back


class UploadManager:

    # ─────────────────────────────────────────────────────────────────────────
    # SAVE
    # ─────────────────────────────────────────────────────────────────────────

    def save_upload(self, file_storage) -> tuple:
        """
        Validates an uploaded CSV file (a Werkzeug FileStorage) and returns
        (summary_dict, DataFrame). Raises ValueError with a clear message
        on any validation failure.

        Returns the DataFrame directly rather than persisting it and
        having the caller read it back — POST /api/ai/train's background
        job needs the exact data THIS upload validated, and a two-step
        save-then-reread would leave a gap for a second admin's concurrent
        upload to land in and silently hand the wrong data to the first
        job's training (confirmed to actually reproduce, not just
        theorised, during Stage C testing). Returning it from the same
        call that validated it closes that gap entirely: there's no
        "in between" for another request to interleave into.
        """
        filename = getattr(file_storage, 'filename', '') or ''
        if not filename.lower().endswith('.csv'):
            raise ValueError("Only .csv files are accepted.")

        try:
            df = pd.read_csv(file_storage)
        except Exception as exc:
            raise ValueError(f"Could not parse CSV: {exc}")

        if df.empty:
            raise ValueError("CSV file is empty.")
        if len(df) > _MAX_ROWS:
            raise ValueError(f"CSV has {len(df)} rows — maximum allowed is {_MAX_ROWS}.")

        df = self._normalise_columns(df)

        missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required column(s): {', '.join(missing)}. "
                f"Found columns: {', '.join(df.columns)}"
            )

        # Validate the date column actually parses
        parsed_dates = pd.to_datetime(df['date'], errors='coerce')
        bad_dates = int(parsed_dates.isna().sum())
        if bad_dates == len(df):
            raise ValueError("No rows had a parseable 'date' value.")
        df = df[parsed_dates.notna()].copy()
        df['date'] = parsed_dates[parsed_dates.notna()]

        # SRS 5.6.7 — exclude rows older than the 24-month (2-year) training window.
        # Must be visible to the admin, not silently dropped (see §2.5 lesson).
        cutoff = pd.Timestamp.now().normalize() - pd.DateOffset(months=_MAX_MONTHS_BACK)
        too_old = df['date'] < cutoff
        skipped_too_old = int(too_old.sum())
        if skipped_too_old:
            logger.info(
                f"Excluding {skipped_too_old} row(s) older than the "
                f"{_MAX_MONTHS_BACK}-month cap (cutoff {cutoff.date().isoformat()})"
            )
        df = df[~too_old].copy()
        if df.empty:
            raise ValueError(
                f"All rows fall outside the {_MAX_MONTHS_BACK}-month (2-year) data "
                f"window allowed by SRS 5.6.7 (cutoff {cutoff.date().isoformat()})."
            )

        has_gps = 'latitude' in df.columns and 'longitude' in df.columns
        if has_gps:
            df['latitude']  = pd.to_numeric(df['latitude'],  errors='coerce')
            df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
            out_of_bounds = ~(
                df['latitude'].between(Config.SL_LAT_MIN, Config.SL_LAT_MAX) &
                df['longitude'].between(Config.SL_LNG_MIN, Config.SL_LNG_MAX)
            )
            bad_gps = int((out_of_bounds | df['latitude'].isna() | df['longitude'].isna()).sum())
        else:
            bad_gps = None

        # exchange_area is independent of GPS — a CSV can have either, both, or
        # neither. Not a replacement for the GPS check above.
        has_exchange_area = 'exchange_area' in df.columns
        if has_exchange_area:
            df['exchange_area'] = df['exchange_area'].astype(str).str.strip()
            blank = (df['exchange_area'] == '') | (df['exchange_area'].str.lower() == 'nan')
            bad_exchange_area = int(blank.sum())
        else:
            bad_exchange_area = None

        has_opmc_code = 'opmc_code' in df.columns
        if has_opmc_code:
            df['opmc_code'] = df['opmc_code'].astype(str).str.strip()

        logger.info(f"Validated uploaded training data: {len(df)} rows ({filename})")

        summary = self._summarise(
            df, filename=filename, skipped_bad_dates=bad_dates,
            skipped_bad_gps=bad_gps, skipped_too_old=skipped_too_old,
            skipped_bad_exchange_area=bad_exchange_area,
        )
        return summary, df

    # ─────────────────────────────────────────────────────────────────────────
    # SHAPING — for the two models' expected input
    # ─────────────────────────────────────────────────────────────────────────

    def to_time_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """Group raw upload rows into Prophet's expected [ds, y] daily-count shape."""
        daily = df.groupby(df['date'].dt.date).size().reset_index()
        daily.columns = ['ds', 'y']
        daily['ds'] = pd.to_datetime(daily['ds'])
        daily['y']  = daily['y'].astype(float)
        return daily.sort_values('ds').reset_index(drop=True)

    def to_gps_points(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Return [latitude, longitude, category, created_at] for K-Means clustering."""
        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            return None
        out = df.copy()
        if 'category' not in out.columns:
            out['category'] = 'OTHER'
        out['created_at'] = out['date']
        return out[['latitude', 'longitude', 'category', 'created_at']].dropna(
            subset=['latitude', 'longitude']
        )

    def to_exchange_area_groups(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Return [exchange_area, opmc_code, category, created_at] for the
        categorical-grouping clustering fallback (no GPS in the source data —
        see KMeansClusterer.cluster_by_exchange_area()). opmc_code rides along
        as passthrough metadata only; it is not resolved against a real Opmc
        row here (see module docstring — the real export's OPMCCODE values do
        not match this database's Opmc.code values today).
        """
        if 'exchange_area' not in df.columns:
            return None
        out = df.copy()
        if 'category' not in out.columns:
            out['category'] = 'OTHER'
        if 'opmc_code' not in out.columns:
            out['opmc_code'] = None
        out['created_at'] = out['date']
        out['exchange_area'] = out['exchange_area'].astype(str).str.strip()
        blank = (out['exchange_area'] == '') | (out['exchange_area'].str.lower() == 'nan')
        return out.loc[~blank, ['exchange_area', 'opmc_code', 'category', 'created_at']]

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE
    # ─────────────────────────────────────────────────────────────────────────

    def _normalise_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        rename_map = {}
        for col in df.columns:
            key = col.strip().lower()
            canonical = _COLUMN_ALIASES.get(key, key)
            rename_map[col] = canonical
        return df.rename(columns=rename_map)

    def _summarise(self, df: pd.DataFrame, filename: str = None,
                    skipped_bad_dates: int = 0, skipped_bad_gps: Optional[int] = None,
                    skipped_too_old: int = 0,
                    skipped_bad_exchange_area: Optional[int] = None) -> dict:
        has_gps           = 'latitude' in df.columns and 'longitude' in df.columns
        has_exchange_area = 'exchange_area' in df.columns
        # GPS wins when a CSV somehow has both (app.py's _run_training_job mirrors
        # this priority) — real coordinates are strictly more informative than a
        # categorical code, so there's no reason to prefer the fallback over them.
        clustering_method = 'kmeans' if has_gps else ('categorical' if has_exchange_area else None)
        return {
            'filename':          filename,
            'rowCount':          len(df),
            'dateRange': {
                'from': df['date'].min().date().isoformat() if len(df) else None,
                'to':   df['date'].max().date().isoformat() if len(df) else None,
            },
            'hasGpsColumns':          has_gps,
            'hasExchangeAreaColumn':  has_exchange_area,
            'hasOpmcCodeColumn':      'opmc_code' in df.columns,
            'hasCategoryColumn':      'category' in df.columns,
            'skippedRows': {
                'badDates':       skipped_bad_dates,
                'badGps':         skipped_bad_gps,
                'badExchangeArea': skipped_bad_exchange_area,
                'tooOld':         skipped_too_old,
            },
            'maxTrainingWindowMonths': _MAX_MONTHS_BACK,
            'usableForForecasting': True,
            'usableForClustering':  bool(has_gps or has_exchange_area),
            'clusteringMethod':     clustering_method,
        }
