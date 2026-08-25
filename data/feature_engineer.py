"""
data/feature_engineer.py — Feature Engineering
================================================
Creates additional regressors and features to improve Prophet accuracy
and enrich the K-Means clustering and route optimisation inputs.

Features created:
  - Calendar features: day_of_week, is_weekend, month, week_of_year
  - Sri Lanka public holidays (as binary regressor for Prophet)
  - Seasonal flags: monsoon, post-monsoon
  - Rolling statistics: 7-day, 30-day rolling mean
  - Lag features: fault_lag_7, fault_lag_14 (for trend detection)
  - Distance matrix for Dijkstra graph construction

Usage:
    from data.feature_engineer import FeatureEngineer
    fe = FeatureEngineer()
    df = fe.add_prophet_regressors(df)
    graph = fe.build_distance_graph(tech_df, fault_lat, fault_lng)
"""

import logging
import math
import numpy as np
import pandas as pd
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

from config import Config

logger = logging.getLogger('slt_ai.features')


# ─── Sri Lanka public holidays (static list + rules) ──────────────────────────
SL_PUBLIC_HOLIDAYS_FIXED = {
    (1,  1):  'New Year\'s Day',
    (2,  4):  'Independence Day',
    (5,  1):  'May Day',
    (12, 25): 'Christmas Day',
}

# Variable holidays (approximate dates — update yearly)
SL_VARIABLE_HOLIDAYS_2025 = [
    '2025-01-14',  # Tamil Thai Pongal
    '2025-02-12',  # Maha Sivarathri
    '2025-04-13',  # Sinhala New Year Eve
    '2025-04-14',  # Sinhala and Tamil New Year
    '2025-05-12',  # Vesak Full Moon Poya
    '2025-06-10',  # Eid ul-Fitr
    '2025-08-09',  # Nikini Full Moon Poya
    '2026-01-14',  # Tamil Thai Pongal 2026
    '2026-04-13',  # Sinhala New Year Eve 2026
    '2026-04-14',  # Sinhala and Tamil New Year 2026
]

# Sri Lanka monsoon seasons
MONSOON_MONTHS_SW = [5, 6, 7, 8, 9]         # South-West monsoon (heavy rain)
MONSOON_MONTHS_NE = [11, 12, 1, 2]          # North-East monsoon


class FeatureEngineer:
    """Adds domain-specific features to DataFrames."""

    def __init__(self):
        self._holiday_dates = self._build_holiday_set()

    # ─────────────────────────────────────────────────────────────────────────
    # PROPHET REGRESSORS
    # ─────────────────────────────────────────────────────────────────────────

    def add_prophet_regressors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all external regressors to a Prophet-ready DataFrame.
        Input must have column 'ds' (datetime).

        Added columns:
            is_holiday, is_weekend, month, day_of_week,
            week_of_year, is_sw_monsoon, is_ne_monsoon,
            rolling_mean_7, rolling_mean_30,
            fault_lag_7, fault_lag_14

        Args:
            df: DataFrame with at minimum columns [ds, y].

        Returns:
            df with additional regressor columns.
        """
        df = df.copy()
        df['ds'] = pd.to_datetime(df['ds'])

        # 1. Calendar features
        df['day_of_week'] = df['ds'].dt.dayofweek        # 0=Mon, 6=Sun
        df['is_weekend']  = (df['day_of_week'] >= 5).astype(int)
        df['month']       = df['ds'].dt.month
        df['week_of_year']= df['ds'].dt.isocalendar().week.astype(int)
        df['day_of_month']= df['ds'].dt.day

        # 2. Sri Lanka public holidays
        df['is_holiday'] = df['ds'].apply(
            lambda d: 1 if d.date() in self._holiday_dates else 0
        )

        # 3. Monsoon season flags
        df['is_sw_monsoon'] = df['month'].isin(MONSOON_MONTHS_SW).astype(int)
        df['is_ne_monsoon'] = df['month'].isin(MONSOON_MONTHS_NE).astype(int)

        # 4. Rolling statistics (require sorted df)
        df = df.sort_values('ds').reset_index(drop=True)
        if 'y' in df.columns:
            df['rolling_mean_7']  = (
                df['y'].rolling(window=7,  min_periods=1).mean().round(2)
            )
            df['rolling_mean_30'] = (
                df['y'].rolling(window=30, min_periods=1).mean().round(2)
            )
            # 5. Lag features
            df['fault_lag_7']  = df['y'].shift(7).fillna(0)
            df['fault_lag_14'] = df['y'].shift(14).fillna(0)
        else:
            # Inference mode — no 'y' to compute these from. Leave as NaN so
            # add_regressors_to_future() can fill them with a sensible
            # historical-mean placeholder instead of a literal zero (these
            # are trained as multiplicative regressors strongly correlated
            # with y, so feeding 0 here collapses the forecast toward 0).
            for col in ['rolling_mean_7','rolling_mean_30','fault_lag_7','fault_lag_14']:
                df[col] = np.nan

        logger.debug(f"Added {len(df.columns) - 2} regressor columns")
        return df

    def add_regressors_to_future(
        self,
        future_df: pd.DataFrame,
        historical_mean: float = 0.0
    ) -> pd.DataFrame:
        """
        Add regressors to a Prophet future DataFrame (no 'y' column).

        Args:
            future_df:        DataFrame with 'ds' column only.
            historical_mean:  Mean y from training data (used for lags).

        Returns:
            future_df with regressor columns filled.
        """
        df = self.add_prophet_regressors(future_df)
        # For future dates, lags and rolling means are approximated
        for col in ['rolling_mean_7', 'rolling_mean_30', 'fault_lag_7', 'fault_lag_14']:
            if col in df.columns:
                df[col] = df[col].fillna(historical_mean)
        return df

    # ─────────────────────────────────────────────────────────────────────────
    # CLUSTERING FEATURES
    # ─────────────────────────────────────────────────────────────────────────

    def enrich_gps_for_clustering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enrich GPS DataFrame with density and region context.
        Adds columns useful for evaluating cluster quality.

        Args:
            df: DataFrame with [latitude, longitude, ...].

        Returns:
            df with added columns.
        """
        df = df.copy()

        # Offset from Sri Lanka center (for visual scaling)
        df['lat_offset'] = df['latitude']  - Config.SL_CENTER_LAT
        df['lng_offset'] = df['longitude'] - Config.SL_CENTER_LNG

        # Distance from Colombo (economic activity proxy)
        colombo = Config.SL_DISTRICTS[0]
        df['dist_from_colombo_km'] = df.apply(
            lambda row: haversine_km(
                colombo['lat'], colombo['lng'],
                row['latitude'], row['longitude']
            ),
            axis=1
        ).round(1)

        # Region classification (north/south/east/west)
        df['region_ns'] = df['latitude'].apply(
            lambda lat: 'north' if lat > 8.0 else ('central' if lat > 7.0 else 'south')
        )
        df['region_ew'] = df['longitude'].apply(
            lambda lng: 'west' if lng < 80.5 else ('central' if lng < 81.0 else 'east')
        )

        logger.debug(f"GPS enrichment complete: {len(df)} points")
        return df

    def get_cluster_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract the 2D feature matrix [lat, lng] for K-Means.
        Optionally weighted by distance from centre to improve urban/rural split.

        Args:
            df: DataFrame with [latitude, longitude].

        Returns:
            numpy array of shape (n, 2).
        """
        return df[['latitude', 'longitude']].values

    # ─────────────────────────────────────────────────────────────────────────
    # DIJKSTRA GRAPH CONSTRUCTION
    # ─────────────────────────────────────────────────────────────────────────

    def build_distance_graph(
        self,
        technicians_df: pd.DataFrame,
        fault_lat: float,
        fault_lng: float
    ) -> Dict:
        """
        Build a weighted graph for Dijkstra's algorithm.

        Nodes: technician GPS positions + fault position
        Edges: Haversine distances (km) between all nodes

        Args:
            technicians_df: DataFrame with [technician_id, latitude, longitude, status, ...].
            fault_lat:      Target fault latitude.
            fault_lng:      Target fault longitude.

        Returns:
            dict: {
                'nodes':  { node_id: {'lat': float, 'lng': float, 'type': str} },
                'edges':  { node_id: [(neighbour_id, weight_km), ...] },
                'fault_node': 'fault_0'
            }
        """
        nodes  = {}
        edges  = {}

        # Add fault as node 'fault_0'
        nodes['fault_0'] = {
            'lat':  fault_lat,
            'lng':  fault_lng,
            'type': 'fault'
        }
        edges['fault_0'] = []

        # Add each technician as a node
        for _, row in technicians_df.iterrows():
            nid = f"tech_{row['technician_id']}"
            nodes[nid] = {
                'lat':            float(row['latitude']),
                'lng':            float(row['longitude']),
                'type':           'technician',
                'technician_id':  row['technician_id'],
                'full_name':      row.get('full_name', ''),
                'phone':          row.get('phone', ''),
                'status':         row.get('status', 'UNKNOWN'),
                'current_job_id': row.get('current_job_id'),
            }
            edges[nid] = []

        # Build edges (fully connected graph — every node connects to every other)
        node_ids = list(nodes.keys())
        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                a_id, b_id = node_ids[i], node_ids[j]
                a, b = nodes[a_id], nodes[b_id]
                dist = haversine_km(a['lat'], a['lng'], b['lat'], b['lng'])
                edges[a_id].append((b_id, round(dist, 3)))
                edges[b_id].append((a_id, round(dist, 3)))

        logger.debug(
            f"Built distance graph: {len(nodes)} nodes, "
            f"{sum(len(v) for v in edges.values()) // 2} edges"
        )
        return {
            'nodes':      nodes,
            'edges':      edges,
            'fault_node': 'fault_0'
        }

    # ─────────────────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _build_holiday_set(self) -> set:
        """Build a set of date objects for all known SL holidays."""
        holidays = set()

        # Variable holidays
        for d_str in SL_VARIABLE_HOLIDAYS_2025:
            try:
                holidays.add(datetime.strptime(d_str, '%Y-%m-%d').date())
            except ValueError:
                pass

        # Fixed holidays — generate for 2024–2027
        for year in range(2024, 2028):
            for (month, day), _ in SL_PUBLIC_HOLIDAYS_FIXED.items():
                try:
                    holidays.add(date(year, month, day))
                except ValueError:
                    pass

        logger.debug(f"Loaded {len(holidays)} Sri Lanka public holidays")
        return holidays


# ─── Standalone utility functions ─────────────────────────────────────────────

def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """
    Calculate the great-circle distance in kilometres between two GPS points
    using the Haversine formula.

    Args:
        lat1, lng1: Source coordinates (degrees).
        lat2, lng2: Destination coordinates (degrees).

    Returns:
        Distance in kilometres (float).
    """
    R = 6371.0  # Earth radius in km

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi       = math.radians(lat2 - lat1)
    dlambda    = math.radians(lng2 - lng1)

    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def estimate_travel_minutes(distance_km: float, avg_speed_kmh: float = 40.0) -> int:
    """
    Estimate travel time in minutes given distance and average speed.
    Default speed 40 km/h (realistic for Sri Lanka road conditions).

    Args:
        distance_km:   Distance in kilometres.
        avg_speed_kmh: Average vehicle speed (default 40 km/h for SL roads).

    Returns:
        Estimated travel time in minutes (int).
    """
    if distance_km <= 0 or avg_speed_kmh <= 0:
        return 0
    return max(1, round((distance_km / avg_speed_kmh) * 60))
