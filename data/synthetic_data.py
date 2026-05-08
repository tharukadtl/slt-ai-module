"""
data/synthetic_data.py — Synthetic / Fallback Data Generator
=============================================================
Generates realistic synthetic data for all AI models when the MySQL
database is unavailable (e.g., during development, testing, or demo).

All synthetic data is seeded for reproducibility and mirrors the
statistical patterns expected from 12–24 months of real SLT fault data.

Usage:
    from data.synthetic_data import SyntheticDataGenerator
    gen = SyntheticDataGenerator(seed=42)
    ts_df   = gen.fault_time_series(days=365)
    gps_df  = gen.fault_gps_points(n=800)
    tech_df = gen.technician_locations(n=25)
"""

import logging
import random
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
from typing import List, Optional

from config import Config

logger = logging.getLogger('slt_ai.synthetic')


class SyntheticDataGenerator:
    """
    Generates statistically realistic synthetic data for Sri Lanka Telecom
    field operations. Seeded for reproducibility.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng  = np.random.default_rng(seed)
        random.seed(seed)
        logger.info(f"SyntheticDataGenerator initialised (seed={seed})")

    # ─────────────────────────────────────────────────────────────────────────
    # TIME-SERIES DATA (Prophet training)
    # ─────────────────────────────────────────────────────────────────────────

    def fault_time_series(
        self,
        days: int = 540,
        base_faults_per_day: float = 18.0
    ) -> pd.DataFrame:
        """
        Generate daily fault count time-series.

        Incorporates realistic patterns:
        - Weekly seasonality (more faults Mon–Fri, fewer weekends)
        - Yearly seasonality (monsoon spikes in May–Sept and Nov–Jan)
        - Upward trend (system growth ~8% per year)
        - Random noise

        Args:
            days:                  Number of days to generate.
            base_faults_per_day:   Baseline daily fault count.

        Returns:
            DataFrame with columns [ds, y] ready for Prophet.
        """
        end_date   = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
        start_date = end_date - timedelta(days=days - 1)
        dates      = pd.date_range(start=start_date, periods=days, freq='D')

        counts = []
        for i, d in enumerate(dates):

            # 1. Linear growth trend: ~8% per year
            trend = 1.0 + (i / 365) * 0.08

            # 2. Weekly seasonality: Mon(1.2)→Fri(1.1)→Sat(0.6)→Sun(0.5)
            dow_factors = [1.2, 1.15, 1.1, 1.1, 1.1, 0.6, 0.5]
            weekly      = dow_factors[d.dayofweek]

            # 3. Monthly seasonality — Sri Lanka rainfall & usage patterns
            monthly_factors = {
                1: 1.15,  # Jan — NE monsoon, heavy rain = more outages
                2: 1.05,
                3: 0.95,
                4: 1.10,  # Apr — pre-SW monsoon instability
                5: 1.25,  # May — SW monsoon starts, peak faults
                6: 1.30,  # Jun — peak SW monsoon
                7: 1.25,
                8: 1.20,
                9: 1.10,  # Sep — SW monsoon tapering
                10: 0.95,
                11: 1.10, # Nov — NE monsoon starts
                12: 1.20, # Dec — NE monsoon peak
            }
            monthly = monthly_factors.get(d.month, 1.0)

            # 4. Public holiday effect (fewer reported faults)
            is_holiday = (
                (d.month == 1  and d.day == 1 ) or  # New Year
                (d.month == 2  and d.day == 4 ) or  # Independence Day
                (d.month == 4  and d.day in [13, 14]) or  # SL New Year
                (d.month == 5  and d.day == 1 ) or  # May Day
                (d.month == 12 and d.day == 25)     # Christmas
            )
            holiday_factor = 0.55 if is_holiday else 1.0

            # 5. Combine all factors
            expected = base_faults_per_day * trend * weekly * monthly * holiday_factor

            # 6. Add Poisson noise (count data)
            count = int(self.rng.poisson(max(expected, 1.0)))
            counts.append(count)

        df = pd.DataFrame({'ds': dates, 'y': counts})
        df['ds'] = pd.to_datetime(df['ds'])
        logger.info(f"Generated {len(df)} days of synthetic fault time-series")
        return df

    # ─────────────────────────────────────────────────────────────────────────
    # GPS FAULT POINTS (K-Means clustering)
    # ─────────────────────────────────────────────────────────────────────────

    def fault_gps_points(
        self,
        n: int = 1000,
        days_back: int = 180
    ) -> pd.DataFrame:
        """
        Generate synthetic fault GPS coordinates for Sri Lanka.

        Points are clustered around realistic high-demand zones:
        - Colombo and suburbs (40% of faults)
        - Kandy (15%)
        - Galle and Southern (12%)
        - Northern (Jaffna area, 10%)
        - Eastern (Batticaloa, 8%)
        - Central / Hill country (8%)
        - Other scattered (7%)

        Args:
            n:          Total number of fault GPS points.
            days_back:  Date range span for created_at column.

        Returns:
            DataFrame with [id, latitude, longitude, category, status, priority, created_at].
        """
        # Cluster centres and their weights / spread
        clusters = [
            {'centre': (6.9271, 79.8612), 'weight': 0.40, 'spread': 0.18,
             'name': 'Colombo Metro'},
            {'centre': (7.2906, 80.6337), 'weight': 0.15, 'spread': 0.15,
             'name': 'Kandy'},
            {'centre': (6.0535, 80.2210), 'weight': 0.12, 'spread': 0.12,
             'name': 'Galle'},
            {'centre': (9.6615, 80.0255), 'weight': 0.10, 'spread': 0.20,
             'name': 'Jaffna'},
            {'centre': (7.7170, 81.6924), 'weight': 0.08, 'spread': 0.18,
             'name': 'Batticaloa'},
            {'centre': (7.8731, 80.7718), 'weight': 0.08, 'spread': 0.25,
             'name': 'Central'},
            {'centre': (7.4818, 80.3609), 'weight': 0.07, 'spread': 0.30,
             'name': 'Scattered'},
        ]

        rows = []
        end_dt    = datetime.today()
        start_dt  = end_dt - timedelta(days=days_back)
        total_sec = int((end_dt - start_dt).total_seconds())

        for i in range(n):
            # Choose cluster by weight
            cluster = self.rng.choice(
                clusters,
                p=[c['weight'] for c in clusters]
            )
            clat, clng = cluster['centre']
            spread     = cluster['spread']

            # Sample GPS from bivariate normal around cluster centre
            lat = float(self.rng.normal(clat, spread * 0.5))
            lng = float(self.rng.normal(clng, spread * 0.7))

            # Clip to Sri Lanka bounds
            lat = float(np.clip(lat, Config.SL_LAT_MIN, Config.SL_LAT_MAX))
            lng = float(np.clip(lng, Config.SL_LNG_MIN, Config.SL_LNG_MAX))

            # Random created_at within range
            offset = timedelta(seconds=int(self.rng.integers(0, total_sec)))
            created_at = start_dt + offset

            rows.append({
                'id':         i + 1,
                'latitude':   round(lat, 6),
                'longitude':  round(lng, 6),
                'category':   self.rng.choice(Config.FAULT_CATEGORIES,
                                               p=[0.40, 0.30, 0.15, 0.10, 0.05]),
                'status':     self.rng.choice(
                                  ['COMPLETED', 'CANCELLED', 'OPEN'],
                                  p=[0.70, 0.15, 0.15]
                              ),
                'priority':   self.rng.choice(['HIGH','MEDIUM','LOW'], p=[0.20,0.55,0.25]),
                'created_at': created_at,
                'branch_id':  int(self.rng.integers(1, 6)),
            })

        df = pd.DataFrame(rows)
        logger.info(f"Generated {len(df)} synthetic fault GPS points")
        return df

    # ─────────────────────────────────────────────────────────────────────────
    # TECHNICIAN LOCATIONS (Route optimiser)
    # ─────────────────────────────────────────────────────────────────────────

    def technician_locations(self, n: int = 25) -> pd.DataFrame:
        """
        Generate synthetic technician current GPS positions.

        Technicians are spread across Sri Lanka, mostly clustered
        around urban centres and their branch offices.

        Args:
            n: Number of technicians to generate.

        Returns:
            DataFrame with [technician_id, full_name, phone, branch_id,
                            latitude, longitude, status, last_seen,
                            current_job_id].
        """
        tech_areas = [
            (6.9271, 79.8612, 1, 0.15),   # Colombo branch
            (7.2906, 80.6337, 2, 0.10),   # Kandy branch
            (6.0535, 80.2210, 3, 0.10),   # Galle branch
            (9.6615, 80.0255, 4, 0.12),   # Jaffna branch
            (7.7170, 81.6924, 5, 0.12),   # Batticaloa branch
        ]

        statuses = ['AVAILABLE','IN_PROGRESS','TRAVELLING','PAUSED','OFFLINE']
        status_p = [0.40, 0.30, 0.15, 0.05, 0.10]

        sl_names = [
            'Kasun Perera','Nuwan Silva','Chamara Fernando','Dilshan Jayawardena',
            'Pradeep Rathnayake','Samantha Gunasekara','Ruwan Wickramasinghe',
            'Harsha Bandara','Tharaka Dissanayake','Lahiru Senanayake',
            'Chathura Rajapaksha','Nalin Karunaratne','Buddhika Samarasinghe',
            'Isuru Madushanka','Malindu Kumara','Sanjeewa Hewavitharana',
            'Asanka Pathirana','Gayan Amarasekara','Niroshan Mendis','Sujeewa Weerasinghe',
            'Damith Liyanage','Kavinda Jayasuriya','Chanaka Kumarasinghe',
            'Malith Abeysinghe','Ashan Ranasinghe',
        ]

        rows = []
        now  = datetime.now()

        for i in range(min(n, len(sl_names))):
            # Choose branch area
            area_idx = i % len(tech_areas)
            clat, clng, branch_id, spread = tech_areas[area_idx]

            # GPS offset from branch centre
            lat = float(np.clip(
                self.rng.normal(clat, spread),
                Config.SL_LAT_MIN, Config.SL_LAT_MAX
            ))
            lng = float(np.clip(
                self.rng.normal(clng, spread),
                Config.SL_LNG_MIN, Config.SL_LNG_MAX
            ))

            status = str(self.rng.choice(statuses, p=status_p))

            # Active techs have recent location update
            mins_ago = int(self.rng.integers(1, 90))
            last_seen = now - timedelta(minutes=mins_ago)

            # Some techs have an active job
            current_job_id = None
            if status in ('IN_PROGRESS', 'TRAVELLING'):
                current_job_id = int(self.rng.integers(100, 9999))

            rows.append({
                'technician_id':  i + 1,
                'full_name':      sl_names[i],
                'phone':          f'07{self.rng.integers(10000000, 99999999):08d}',
                'branch_id':      branch_id,
                'branch_name':    ['Colombo','Kandy','Galle','Jaffna','Batticaloa'][branch_id-1],
                'latitude':       round(lat, 6),
                'longitude':      round(lng, 6),
                'status':         status,
                'last_seen':      last_seen.isoformat(),
                'current_job_id': current_job_id,
                'avatar_initial': sl_names[i][0].upper(),
            })

        df = pd.DataFrame(rows)
        logger.info(f"Generated {len(df)} synthetic technician locations")
        return df

    # ─────────────────────────────────────────────────────────────────────────
    # BRANCHES
    # ─────────────────────────────────────────────────────────────────────────

    def branches(self) -> pd.DataFrame:
        """Return synthetic branch reference data."""
        data = [
            {'id':1, 'name':'Colombo',     'region':'Western',    'code':'COL', 'lat':6.9271, 'lng':79.8612},
            {'id':2, 'name':'Kandy',       'region':'Central',    'code':'KAN', 'lat':7.2906, 'lng':80.6337},
            {'id':3, 'name':'Galle',       'region':'Southern',   'code':'GAL', 'lat':6.0535, 'lng':80.2210},
            {'id':4, 'name':'Jaffna',      'region':'Northern',   'code':'JAF', 'lat':9.6615, 'lng':80.0255},
            {'id':5, 'name':'Batticaloa',  'region':'Eastern',    'code':'BAT', 'lat':7.7170, 'lng':81.6924},
        ]
        return pd.DataFrame(data)

    # ─────────────────────────────────────────────────────────────────────────
    # COMBINED: get all data needed by AI pipeline
    # ─────────────────────────────────────────────────────────────────────────

    def all_data(self, forecast_days: int = 540) -> dict:
        """
        Return a dict of all synthetic data frames.
        Convenience method used by models when DB is unavailable.

        Returns:
            {
                'time_series':    DataFrame [ds, y],
                'fault_gps':      DataFrame [id, latitude, longitude, ...],
                'technicians':    DataFrame [technician_id, latitude, longitude, ...],
                'branches':       DataFrame,
            }
        """
        return {
            'time_series':  self.fault_time_series(days=forecast_days),
            'fault_gps':    self.fault_gps_points(n=1000),
            'technicians':  self.technician_locations(n=25),
            'branches':     self.branches(),
        }
