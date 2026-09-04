"""
tests/test_resource_planner.py — Predictive Resource Planning (FR-33) Unit Tests
=================================================================================
Verifies the actual math and behaviour of FR-33 Stage 2 (models/resource_planner.py
+ KMeansClusterer.assign_zones + GET /api/ai/resource-plan), not just that the
code paths run without raising:

  1. Zone assignment      — assign_zones() maps known coordinates to their
                             nearest fitted centroid; raises before any fit.
  2. Zone/shift ratios    — hotspot breakdown reflects known, controlled
                             fault-percent / shift-percent inputs exactly,
                             including the zero-historical-shift-data ->
                             even 1/3 split fallback.
  3. Suggestion math      — Technician count (ceil of predicted/avg-jobs),
                             Vehicle count (1:1 with Technician), Material
                             shortfalls (summed across the horizon vs. stock,
                             flagged only when actually insufficient).
  4. Insufficient history — the shared 6-month rule, exercised through
                             /api/ai/resource-plan specifically (not just
                             /api/ai/predictions or /api/ai/clusters).
  5. Endpoint validation  — horizon clamping/fallback, sort order, wrong
                             method, planner-unavailable.

Run:
    cd slt-ai-module
    python -m pytest tests/test_resource_planner.py -v
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from models.resource_planner import ResourcePlanner, SHIFT_WINDOWS


# ═════════════════════════════════════════════════════════════════════════════
# FAKES — deterministic stand-ins for the forecaster/clusterer/extractor so
# distribution math can be checked against known ratios, not just "ran OK".
# ═════════════════════════════════════════════════════════════════════════════

class FakeForecaster:
    def __init__(self, forecast_days):
        self._forecast_days = forecast_days

    def forecast(self, df, horizon=7):
        return {'insufficientData': False, 'forecast': self._forecast_days}


class FakeClusterer:
    def __init__(self, clusters, total_faults, zone_ids=None):
        self._clusters = clusters
        self._total_faults = total_faults
        self._zone_ids = zone_ids or []

    def cluster(self, gps_df, n_clusters=None):
        return {'clusters': self._clusters, 'totalFaults': self._total_faults}

    def assign_zones(self, df):
        return np.array(self._zone_ids[:len(df)])


class FakeExtractor:
    def __init__(self, shift_df=None, tech_shift_df=None, usage_df=None, stock_df=None):
        self._shift_df = shift_df
        self._tech_shift_df = tech_shift_df
        self._usage_df = usage_df
        self._stock_df = stock_df

    def get_fault_shift_distribution(self, days_back=180):
        return self._shift_df

    def get_technician_shift_performance(self, days_back=180):
        return self._tech_shift_df

    def get_material_usage_with_location(self, days_back=365):
        return self._usage_df

    def get_material_stock(self):
        return self._stock_df


def make_planner(forecaster=None, clusterer=None, extractor=None):
    return ResourcePlanner(forecaster, clusterer, extractor)


# ═════════════════════════════════════════════════════════════════════════════
# 1. ZONE ASSIGNMENT — KMeansClusterer.assign_zones()
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope='module')
def zone_assignment_synth():
    from data.synthetic_data import SyntheticDataGenerator
    return SyntheticDataGenerator(seed=42)


@pytest.fixture(scope='module')
def fitted_clusterer(zone_assignment_synth):
    from models.clustering import KMeansClusterer
    kc = KMeansClusterer()
    gps = zone_assignment_synth.fault_gps_points(n=300)
    result = kc.cluster(gps, n_clusters=4)
    return kc, result


class TestZoneAssignment:

    def test_assign_zones_maps_centroids_to_own_cluster(self, fitted_clusterer):
        """
        A point placed EXACTLY at a cluster's own centroid must be assigned
        to that cluster — distance 0 from its own centroid can't be beaten
        by any other (distinct) centroid.
        """
        kc, result = fitted_clusterer
        clusters = result['clusters']
        query = pd.DataFrame({
            'latitude':  [c['centroid']['lat'] for c in clusters],
            'longitude': [c['centroid']['lng'] for c in clusters],
        })
        assigned = [int(x) for x in kc.assign_zones(query)]
        expected = [int(c['clusterId']) for c in clusters]
        assert assigned == expected

    def test_assign_zones_matches_independent_nearest_centroid(self, fitted_clusterer):
        """
        For arbitrary points, assign_zones()'s label must match an
        independently-computed nearest-centroid lookup — plain Euclidean
        distance on raw lat/lng, the same space K-Means was fit on per
        assign_zones()'s own docstring (no scaling step involved).
        """
        kc, result = fitted_clusterer
        clusters = result['clusters']
        rng = np.random.default_rng(7)
        lats = rng.uniform(Config.SL_LAT_MIN, Config.SL_LAT_MAX, size=25)
        lngs = rng.uniform(Config.SL_LNG_MIN, Config.SL_LNG_MAX, size=25)
        query = pd.DataFrame({'latitude': lats, 'longitude': lngs})

        assigned = [int(x) for x in kc.assign_zones(query)]

        for i in range(len(query)):
            dists = {
                int(c['clusterId']): (c['centroid']['lat'] - lats[i]) ** 2
                                    + (c['centroid']['lng'] - lngs[i]) ** 2
                for c in clusters
            }
            expected_id = min(dists, key=dists.get)
            assert assigned[i] == expected_id, (
                f"Point ({lats[i]:.4f},{lngs[i]:.4f}) assigned to "
                f"{assigned[i]}, nearest centroid is {expected_id}"
            )

    def test_assign_zones_raises_before_any_fit(self):
        """
        A KMeansClusterer that has never had cluster() run (and has no
        saved model loaded) must raise, not silently predict against an
        unfit/absent model.
        """
        from models.clustering import KMeansClusterer
        kc = KMeansClusterer()
        kc._trained = False
        kc._model = None
        with pytest.raises(RuntimeError):
            kc.assign_zones(pd.DataFrame({'latitude': [6.93], 'longitude': [79.86]}))

    def test_assign_zones_empty_input_returns_empty_array(self, fitted_clusterer):
        kc, _ = fitted_clusterer
        result = kc.assign_zones(pd.DataFrame({'latitude': [], 'longitude': []}))
        assert len(result) == 0


# ═════════════════════════════════════════════════════════════════════════════
# 2. ZONE / SHIFT DISTRIBUTION MATH
# ═════════════════════════════════════════════════════════════════════════════

class TestShiftDistributionNormalisation:

    def test_known_ratios_are_preserved(self):
        planner = make_planner()
        shift_df = pd.DataFrame({
            'shift':       ['MORNING', 'AFTERNOON', 'EVENING'],
            'fault_count': [50, 30, 20],
        })
        pct, source = planner._normalise_shift_distribution(shift_df)
        assert pct['MORNING']   == pytest.approx(0.5)
        assert pct['AFTERNOON'] == pytest.approx(0.3)
        assert pct['EVENING']   == pytest.approx(0.2)
        assert sum(pct.values()) == pytest.approx(1.0)
        assert source == 'historical'

    def test_missing_shift_defaults_to_zero_not_dropped(self):
        """A shift with zero historical rows is 0.0, not silently absent."""
        planner = make_planner()
        shift_df = pd.DataFrame({'shift': ['MORNING', 'AFTERNOON'], 'fault_count': [80, 20]})
        pct, source = planner._normalise_shift_distribution(shift_df)
        assert pct['EVENING'] == 0.0
        assert set(pct.keys()) == set(SHIFT_WINDOWS)
        # A real (if partial) historical distribution -- still 'historical', not a fallback.
        assert source == 'historical'

    def test_none_falls_back_to_even_split(self):
        planner = make_planner()
        pct, source = planner._normalise_shift_distribution(None)
        for s in SHIFT_WINDOWS:
            assert pct[s] == pytest.approx(1 / 3)
        assert source == 'even_split_fallback'

    def test_empty_dataframe_falls_back_to_even_split(self):
        planner = make_planner()
        pct, source = planner._normalise_shift_distribution(pd.DataFrame(columns=['shift', 'fault_count']))
        for s in SHIFT_WINDOWS:
            assert pct[s] == pytest.approx(1 / 3)
        assert source == 'even_split_fallback'

    def test_zero_total_falls_back_to_even_split(self):
        """All-zero counts (rows exist but nothing happened) is the same
        degenerate case as no data at all."""
        planner = make_planner()
        shift_df = pd.DataFrame({'shift': SHIFT_WINDOWS, 'fault_count': [0, 0, 0]})
        pct, source = planner._normalise_shift_distribution(shift_df)
        for s in SHIFT_WINDOWS:
            assert pct[s] == pytest.approx(1 / 3)
        assert source == 'even_split_fallback'


class TestShiftDistributionSourceSignal:
    """
    The transparency signal itself: GET /api/ai/resource-plan's real output
    (generate_plan(), not the private helper in isolation) must carry
    shiftDistributionSource so a caller (the Web Admin Resource Planning
    page) can tell a genuine historical-ratio prediction apart from the
    even-1/3 fallback default -- the same purpose insufficientData already
    serves for the forecast half of this same response. Both states proven
    end-to-end through the real combiner, not just the private helper.
    """

    FORECAST_DAYS = [{'ds': '2026-01-01', 'yhat': 90.0}]
    ZONES = [{'clusterId': 0, 'regionName': 'Zone A', 'faultPercent': 100.0}]

    def _plan(self, shift_df):
        forecaster = FakeForecaster(self.FORECAST_DAYS)
        clusterer  = FakeClusterer(self.ZONES, total_faults=90)
        extractor  = FakeExtractor(shift_df=shift_df)
        planner    = ResourcePlanner(forecaster, clusterer, extractor)
        return planner.generate_plan(pd.DataFrame(), pd.DataFrame(), horizon_days=1)

    def test_real_historical_shift_data_reports_historical(self):
        shift_df = pd.DataFrame({
            'shift':       ['MORNING', 'AFTERNOON', 'EVENING'],
            'fault_count': [50, 30, 20],
        })
        result = self._plan(shift_df)
        assert result['insufficientData'] is False
        assert result['shiftDistributionSource'] == 'historical'
        # Not just the field's presence -- the hotspot math it's describing really did use the
        # real 50/30/20 ratio, not a silent 1/3 split (would be ~30.0 for every shift if it had).
        by_shift = {h['shift']: h['predictedFaultCount'] for h in result['hotspots']}
        assert by_shift['MORNING']   == pytest.approx(90.0 * 0.5)
        assert by_shift['AFTERNOON'] == pytest.approx(90.0 * 0.3)
        assert by_shift['EVENING']   == pytest.approx(90.0 * 0.2)

    def test_no_historical_shift_data_reports_even_split_fallback(self):
        result = self._plan(None)
        assert result['insufficientData'] is False
        assert result['shiftDistributionSource'] == 'even_split_fallback'
        # And the math really did fall back to an even split -- every shift gets the same count.
        by_shift = {h['shift']: h['predictedFaultCount'] for h in result['hotspots']}
        for s in SHIFT_WINDOWS:
            assert by_shift[s] == pytest.approx(90.0 / 3)

    def test_all_zero_historical_counts_also_reports_fallback(self):
        """Rows exist but nothing happened -- the same degenerate case as no data at all,
        and the signal must say so, not report 'historical' for a distribution that's really
        just the fallback's own numbers by coincidence."""
        shift_df = pd.DataFrame({'shift': SHIFT_WINDOWS, 'fault_count': [0, 0, 0]})
        result = self._plan(shift_df)
        assert result['shiftDistributionSource'] == 'even_split_fallback'


class TestHotspotZoneShiftBreakdown:
    """
    Verifies _build_hotspots (and generate_plan end-to-end) actually
    distributes the forecast total according to known zone/shift ratios,
    not merely that it returns *a* value for each combination.
    """

    ZONES = [
        {'clusterId': 0, 'regionName': 'Zone A', 'faultPercent': 70.0},
        {'clusterId': 1, 'regionName': 'Zone B', 'faultPercent': 30.0},
    ]
    SHIFT_PCT = {'MORNING': 0.5, 'AFTERNOON': 0.3, 'EVENING': 0.2}

    def test_build_hotspots_reflects_known_zone_and_shift_ratios(self):
        planner = make_planner()
        forecast_days = [{'ds': '2026-01-01', 'yhat': 100.0}]
        hotspots = planner._build_hotspots(
            forecast_days, self.ZONES, self.SHIFT_PCT,
            avg_jobs_per_tech_shift={s: None for s in SHIFT_WINDOWS},
            material_rates={},
        )
        by_key = {(h['zoneId'], h['shift']): h['predictedFaultCount'] for h in hotspots}

        assert by_key[(0, 'MORNING')]   == pytest.approx(100 * 0.70 * 0.5)   # 35.0
        assert by_key[(0, 'AFTERNOON')] == pytest.approx(100 * 0.70 * 0.3)   # 21.0
        assert by_key[(0, 'EVENING')]   == pytest.approx(100 * 0.70 * 0.2)   # 14.0
        assert by_key[(1, 'MORNING')]   == pytest.approx(100 * 0.30 * 0.5)   # 15.0
        assert by_key[(1, 'AFTERNOON')] == pytest.approx(100 * 0.30 * 0.3)   #  9.0
        assert by_key[(1, 'EVENING')]   == pytest.approx(100 * 0.30 * 0.2)   #  6.0

        # Zone-split ratio preserved for a given shift.
        assert by_key[(0, 'MORNING')] / by_key[(1, 'MORNING')] == pytest.approx(70 / 30)
        # Shift-split ratio preserved for a given zone.
        assert by_key[(0, 'MORNING')] / by_key[(0, 'EVENING')] == pytest.approx(0.5 / 0.2)

    def test_generate_plan_end_to_end_reflects_known_ratios(self):
        """Same check, through the actual public entry point with fake
        collaborators standing in for Prophet/K-Means/DB — confirms the
        ratios survive the full orchestration, not just the helper."""
        forecaster = FakeForecaster([{'ds': '2026-02-01', 'yhat': 200.0}])
        clusterer  = FakeClusterer(self.ZONES, total_faults=1000)
        extractor  = FakeExtractor(
            shift_df=pd.DataFrame({'shift': SHIFT_WINDOWS, 'fault_count': [50, 30, 20]}),
        )
        planner = ResourcePlanner(forecaster, clusterer, extractor)

        result = planner.generate_plan(pd.DataFrame(), pd.DataFrame(), horizon_days=1)

        assert result['insufficientData'] is False
        by_key = {(h['zoneId'], h['shift']): h['predictedFaultCount'] for h in result['hotspots']}
        assert by_key[(0, 'MORNING')] == pytest.approx(200 * 0.70 * 0.5)   # 70.0
        assert by_key[(1, 'EVENING')] == pytest.approx(200 * 0.30 * 0.2)   # 12.0

    def test_zero_historical_shift_data_falls_back_to_even_split_end_to_end(self):
        """The zero-historical-shift-data edge case, exercised through the
        real orchestration path: no shift rows at all -> even 1/3 split,
        which shows up as equal predictedFaultCount across shifts for a
        given zone."""
        forecaster = FakeForecaster([{'ds': '2026-02-01', 'yhat': 90.0}])
        clusterer  = FakeClusterer(self.ZONES, total_faults=1000)
        extractor  = FakeExtractor(shift_df=None)   # no historical shift data at all
        planner = ResourcePlanner(forecaster, clusterer, extractor)

        result = planner.generate_plan(pd.DataFrame(), pd.DataFrame(), horizon_days=1)

        zone0 = [h for h in result['hotspots'] if h['zoneId'] == 0]
        counts = {h['shift']: h['predictedFaultCount'] for h in zone0}
        expected = round(90.0 * 0.70 * (1 / 3), 2)
        for s in SHIFT_WINDOWS:
            assert counts[s] == pytest.approx(expected)


# ═════════════════════════════════════════════════════════════════════════════
# 3. TECHNICIAN / VEHICLE / MATERIAL SUGGESTION MATH
# ═════════════════════════════════════════════════════════════════════════════

class TestTechnicianVehicleMaterialSuggestions:

    def test_technician_count_divides_predicted_by_avg_jobs(self):
        planner = make_planner()
        forecast_days = [{'ds': '2026-01-01', 'yhat': 100.0}]
        zones = [{'clusterId': 0, 'regionName': 'Zone A', 'faultPercent': 100.0}]
        shift_pct = {'MORNING': 1.0, 'AFTERNOON': 0.0, 'EVENING': 0.0}
        avg_jobs = {'MORNING': 4.0, 'AFTERNOON': None, 'EVENING': 2.0}

        hotspots = planner._build_hotspots(forecast_days, zones, shift_pct, avg_jobs, material_rates={})
        morning = next(h for h in hotspots if h['shift'] == 'MORNING')

        # predicted = 100*1.0*1.0 = 100 ; 100/4 = 25 exactly -> ceil(25) = 25
        assert morning['predictedFaultCount'] == pytest.approx(100.0)
        assert morning['suggestedTechnicians'] == 25

    def test_technician_count_rounds_up_not_down(self):
        """math.ceil, not integer division — a fractional remainder still
        needs a whole extra technician."""
        planner = make_planner()
        forecast_days = [{'ds': '2026-01-01', 'yhat': 100.0}]
        zones = [{'clusterId': 0, 'regionName': 'Zone A', 'faultPercent': 100.0}]
        shift_pct = {'MORNING': 1.0, 'AFTERNOON': 0.0, 'EVENING': 0.0}
        avg_jobs = {'MORNING': 3.0, 'AFTERNOON': None, 'EVENING': None}

        hotspots = planner._build_hotspots(forecast_days, zones, shift_pct, avg_jobs, material_rates={})
        morning = next(h for h in hotspots if h['shift'] == 'MORNING')
        # predicted = 100 ; 100/3 = 33.33... -> ceil = 34
        assert morning['suggestedTechnicians'] == 34

    def test_no_historical_technician_rate_yields_none_not_a_guess(self):
        planner = make_planner()
        forecast_days = [{'ds': '2026-01-01', 'yhat': 100.0}]
        zones = [{'clusterId': 0, 'regionName': 'Zone A', 'faultPercent': 100.0}]
        shift_pct = {'MORNING': 1.0, 'AFTERNOON': 0.0, 'EVENING': 0.0}
        avg_jobs = {'MORNING': None, 'AFTERNOON': None, 'EVENING': None}

        hotspots = planner._build_hotspots(forecast_days, zones, shift_pct, avg_jobs, material_rates={})
        morning = next(h for h in hotspots if h['shift'] == 'MORNING')
        assert morning['suggestedTechnicians'] is None
        assert morning['suggestedVehicles'] is None

    def test_vehicle_count_matches_technician_count_1_to_1(self):
        """SRS 5.6.8 — Vehicle Suggestion matches Technician count exactly."""
        planner = make_planner()
        forecast_days = [{'ds': '2026-01-01', 'yhat': 123.0}]
        zones = [
            {'clusterId': 0, 'regionName': 'Zone A', 'faultPercent': 60.0},
            {'clusterId': 1, 'regionName': 'Zone B', 'faultPercent': 40.0},
        ]
        shift_pct = {'MORNING': 0.4, 'AFTERNOON': 0.35, 'EVENING': 0.25}
        avg_jobs = {'MORNING': 5.0, 'AFTERNOON': None, 'EVENING': 1.5}

        hotspots = planner._build_hotspots(forecast_days, zones, shift_pct, avg_jobs, material_rates={})
        assert len(hotspots) == 1 * 2 * 3   # days * zones * shifts

        zoneA_morning = next(h for h in hotspots if h['zoneId'] == 0 and h['shift'] == 'MORNING')
        # predicted = 123 * 0.60 * 0.40 = 29.52 ; ceil(29.52 / 5.0) = 6
        assert zoneA_morning['predictedFaultCount'] == pytest.approx(29.52)
        assert zoneA_morning['suggestedTechnicians'] == 6
        assert zoneA_morning['suggestedVehicles'] == 6

        # 1:1 holds across every hotspot generated, whether the count is a
        # concrete int or None (AFTERNOON has no historical rate).
        for h in hotspots:
            assert h['suggestedVehicles'] == h['suggestedTechnicians']

    def test_material_quantity_scales_with_predicted_fault_count(self):
        planner = make_planner()
        forecast_days = [{'ds': '2026-01-01', 'yhat': 100.0}]
        zones = [{'clusterId': 0, 'regionName': 'Zone A', 'faultPercent': 100.0}]
        shift_pct = {'MORNING': 1.0, 'AFTERNOON': 0.0, 'EVENING': 0.0}
        material_rates = {0: {7: {'name': 'Fiber Cable', 'avgQtyPerFault': 2.5}}}

        hotspots = planner._build_hotspots(
            forecast_days, zones, shift_pct,
            avg_jobs_per_tech_shift={s: None for s in SHIFT_WINDOWS},
            material_rates=material_rates,
        )
        morning = next(h for h in hotspots if h['shift'] == 'MORNING')
        material = morning['materials'][0]
        assert material['materialId'] == 7
        assert material['materialName'] == 'Fiber Cable'
        # predicted = 100 ; 100 * 2.5 = 250.0
        assert material['suggestedQuantity'] == pytest.approx(250.0)


class TestMaterialRateComputation:
    """_compute_material_rates: historical avg qty/fault per zone per material."""

    def test_rates_computed_per_zone_from_assigned_usage(self):
        zones = [
            {'clusterId': 0, 'faultPercent': 60.0},
            {'clusterId': 1, 'faultPercent': 40.0},
        ]
        usage_df = pd.DataFrame({
            'latitude':      [6.90, 6.91, 7.29, 7.30],
            'longitude':     [79.86, 79.87, 80.63, 80.64],
            'material_id':   [1, 1, 1, 1],
            'material_name': ['Cable', 'Cable', 'Cable', 'Cable'],
            'quantity_used': [10.0, 20.0, 30.0, 40.0],
        })
        # First two rows -> zone 0, last two -> zone 1.
        clusterer = FakeClusterer(clusters=zones, total_faults=100, zone_ids=[0, 0, 1, 1])
        planner = ResourcePlanner(forecaster=None, clusterer=clusterer, extractor=None)

        rates = planner._compute_material_rates(usage_df, zones, total_faults=100)

        # zone 0: fault_count = 100*0.60=60, qty=10+20=30 -> 30/60=0.5
        assert rates[0][1]['avgQtyPerFault'] == pytest.approx(0.5)
        # zone 1: fault_count = 100*0.40=40, qty=30+40=70 -> 70/40=1.75
        assert rates[1][1]['avgQtyPerFault'] == pytest.approx(1.75)
        assert rates[0][1]['name'] == 'Cable'

    def test_none_usage_returns_empty_rates(self):
        planner = ResourcePlanner(forecaster=None, clusterer=FakeClusterer([], 0), extractor=None)
        assert planner._compute_material_rates(None, [], 0) == {}


class TestAvgJobsPerTechnicianShift:

    def test_pooled_rate_not_mean_of_individual_technician_rates(self):
        """
        Two technicians in the same shift with very different personal
        rates (5.0 vs 0.2 jobs/day) must pool as sum(jobs)/sum(days) = 1.0,
        NOT the unweighted mean of their individual rates (2.6) — robust
        to one technician having far more/fewer days on record.
        """
        planner = make_planner()
        tech_shift_df = pd.DataFrame({
            'shift':          ['MORNING', 'MORNING'],
            'jobs_completed': [10, 2],
            'days_covered':   [2, 10],
        })
        rates = planner._compute_avg_jobs_per_tech_shift(tech_shift_df)
        assert rates['MORNING'] == pytest.approx(1.0)
        assert rates['AFTERNOON'] is None
        assert rates['EVENING'] is None

    def test_none_input_yields_none_for_every_shift(self):
        planner = make_planner()
        rates = planner._compute_avg_jobs_per_tech_shift(None)
        for s in SHIFT_WINDOWS:
            assert rates[s] is None


class TestMaterialShortfalls:

    def test_sums_across_horizon_and_flags_only_actual_shortfalls(self):
        planner = make_planner()
        hotspots = [
            {'materials': [{'materialId': 1, 'materialName': 'Cable', 'suggestedQuantity': 10.0}]},
            {'materials': [{'materialId': 1, 'materialName': 'Cable', 'suggestedQuantity': 15.0}]},
            {'materials': [{'materialId': 2, 'materialName': 'Pole',  'suggestedQuantity': 5.0}]},
            {'materials': [{'materialId': 3, 'materialName': 'Clamp', 'suggestedQuantity': 3.0}]},
        ]
        stock_by_material = {
            1: {'name': 'Cable', 'currentStock': 20.0, 'unit': 'm'},
            2: {'name': 'Pole',  'currentStock': 2.0,  'unit': 'pcs'},
            3: {'name': 'Clamp', 'currentStock': 10.0, 'unit': 'pcs'},
        }
        shortfalls = planner._compute_shortfalls(hotspots, stock_by_material)
        by_id = {s['materialId']: s for s in shortfalls}

        assert by_id[1]['totalSuggestedQuantity'] == pytest.approx(25.0)   # 10+15
        assert by_id[1]['insufficient'] is True                            # 20 < 25
        assert by_id[2]['totalSuggestedQuantity'] == pytest.approx(5.0)
        assert by_id[2]['insufficient'] is True                            # 2 < 5
        assert by_id[3]['totalSuggestedQuantity'] == pytest.approx(3.0)
        assert by_id[3]['insufficient'] is False                           # 10 >= 3

    def test_material_with_no_stock_record_defaults_to_zero_and_flags_short(self):
        planner = make_planner()
        hotspots = [{'materials': [{'materialId': 9, 'materialName': 'Unknown', 'suggestedQuantity': 1.0}]}]
        shortfalls = planner._compute_shortfalls(hotspots, stock_by_material={})
        assert shortfalls[0]['currentStock'] == 0.0
        assert shortfalls[0]['insufficient'] is True

    def test_shortfalls_sorted_by_total_quantity_descending(self):
        planner = make_planner()
        hotspots = [
            {'materials': [
                {'materialId': 1, 'materialName': 'A', 'suggestedQuantity': 5.0},
                {'materialId': 2, 'materialName': 'B', 'suggestedQuantity': 50.0},
            ]},
        ]
        shortfalls = planner._compute_shortfalls(hotspots, stock_by_material={})
        assert [s['materialId'] for s in shortfalls] == [2, 1]


# ═════════════════════════════════════════════════════════════════════════════
# 4 & 5. GET /api/ai/resource-plan — ENDPOINT TESTS
# ═════════════════════════════════════════════════════════════════════════════
# Uses conftest.py's session-scoped `client` fixture (DB mocked unavailable —
# every request runs on synthetic data, same as test_routes.py's other
# endpoint tests) plus its `json_response` / `assert_envelope` helpers.

class TestResourcePlanEndpoint:

    def test_default_returns_200(self, client):
        resp = client.get('/api/ai/resource-plan')
        assert resp.status_code == 200

    def test_returns_success_envelope(self, client, json_response, assert_envelope):
        body = json_response(client.get('/api/ai/resource-plan'))
        assert_envelope(body, success=True)

    def test_data_has_required_keys(self, client, json_response):
        data = json_response(client.get('/api/ai/resource-plan'))['data']
        for key in ['insufficientData', 'horizonDays', 'hotspots', 'materialShortfalls', 'generatedAt']:
            assert key in data, f"Missing data key: {key}"

    def test_default_horizon_is_7(self, client, json_response):
        data = json_response(client.get('/api/ai/resource-plan'))['data']
        assert data['horizonDays'] == 7
        assert len({h['date'] for h in data['hotspots']}) == 7

    def test_horizon_capped_at_90(self, client, json_response):
        """horizon=999 should be clamped to 90."""
        data = json_response(client.get('/api/ai/resource-plan?horizon=999'))['data']
        assert data['horizonDays'] == 90
        assert len({h['date'] for h in data['hotspots']}) == 90

    def test_horizon_floored_at_7(self, client, json_response):
        """horizon=1 should be clamped to 7."""
        data = json_response(client.get('/api/ai/resource-plan?horizon=1'))['data']
        assert data['horizonDays'] == 7
        assert len({h['date'] for h in data['hotspots']}) == 7

    def test_non_numeric_horizon_falls_back_to_default(self, client, json_response):
        """_int_param() swallows a bad value to the default rather than 500ing."""
        resp = client.get('/api/ai/resource-plan?horizon=abc')
        assert resp.status_code == 200
        assert json_response(resp)['data']['horizonDays'] == 7

    def test_predicted_fault_counts_non_negative(self, client, json_response):
        data = json_response(client.get('/api/ai/resource-plan'))['data']
        for h in data['hotspots']:
            assert h['predictedFaultCount'] >= 0

    def test_hotspots_sorted_by_predicted_count_descending(self, client, json_response):
        data = json_response(client.get('/api/ai/resource-plan'))['data']
        counts = [h['predictedFaultCount'] for h in data['hotspots']]
        assert counts == sorted(counts, reverse=True)

    def test_wrong_method_returns_405(self, client):
        resp = client.post('/api/ai/resource-plan')
        assert resp.status_code == 405

    def test_unknown_query_param_ignored_not_500(self, client):
        resp = client.get('/api/ai/resource-plan?bogus=xyz')
        assert resp.status_code == 200


class TestResourcePlanUnavailable:

    def test_returns_503_when_planner_not_available(self, client, monkeypatch, json_response, assert_envelope):
        import app as app_mod
        monkeypatch.setattr(app_mod, '_has_resource_planner', False)
        monkeypatch.setattr(app_mod, '_resource_planner', None)

        resp = client.get('/api/ai/resource-plan')
        assert resp.status_code == 503
        assert_envelope(json_response(resp), success=False)


class TestResourcePlanInsufficientHistory:
    """
    FR-33 shares the SRS 5.6.2/5.6.8 six-month minimum-history rule with
    /api/ai/predictions and /api/ai/clusters, but ResourcePlanner.generate_plan()
    has its own insufficientData short-circuit — this must be verified through
    /api/ai/resource-plan itself, not assumed from the other endpoints' coverage.
    """

    def test_insufficient_history_short_circuits_with_empty_hotspots(
        self, client, monkeypatch, json_response, assert_envelope,
    ):
        import app as app_mod

        def fake_forecast(df, horizon=7):
            return {
                'insufficientData': True,
                'historyDays':      42,
                'requiredDays':     Config.FORECAST_MIN_HISTORY_DAYS,
                'forecast':         [],
            }

        monkeypatch.setattr(app_mod._forecaster, 'forecast', fake_forecast)

        resp = client.get('/api/ai/resource-plan?horizon=14')
        assert resp.status_code == 200
        body = json_response(resp)
        assert_envelope(body, success=True)
        data = body['data']
        assert data['insufficientData'] is True
        assert data['historyDays'] == 42
        assert data['requiredDays'] == Config.FORECAST_MIN_HISTORY_DAYS
        assert data['hotspots'] == []
        assert data['materialShortfalls'] == []
        assert 'Insufficient historical data' in body['message']

    def test_clusterer_is_never_reached_when_forecast_is_insufficient(
        self, client, monkeypatch, json_response,
    ):
        """generate_plan() must short-circuit on insufficientData BEFORE
        calling the clusterer at all — verified by making the clusterer
        explode if it's ever invoked."""
        import app as app_mod

        def fake_forecast(df, horizon=7):
            return {'insufficientData': True, 'historyDays': 10, 'requiredDays': 180, 'forecast': []}

        def exploding_cluster(gps_df, n_clusters=None):
            raise AssertionError("clusterer.cluster() should not be called when forecast is insufficient")

        monkeypatch.setattr(app_mod._forecaster, 'forecast', fake_forecast)
        monkeypatch.setattr(app_mod._clusterer, 'cluster', exploding_cluster)

        resp = client.get('/api/ai/resource-plan')
        assert resp.status_code == 200
        assert json_response(resp)['data']['insufficientData'] is True
