"""
models/resource_planner.py — Predictive Resource Planning (FR-33, SRS 5.6.8)
=============================================================================
Combines the fault volume forecast (5.6.2, ProphetForecaster), geographic
demand clusters (5.6.3, KMeansClusterer), and Stage 1's historical-ratio
queries (data_extractor.py) into predicted hotspots — date + shift window +
zone — each with a predicted fault count and suggested Technician / Vehicle /
Material quantities.

This is a combination/orchestration layer, not a new model of its own: it
calls the forecaster and clusterer's already-fitted models directly, the
same way app.py's /api/ai/dashboard combines a "quick forecast" and "cluster
overview". It holds no persisted state of its own (no registry, no versions)
— same shape as DijkstraRouter, which is also a pure-computation class.

Distribution method (both explicit judgement calls, not silently invented):
  - Zone split:  each zone's real historical share of faults, already
                 computed by KMeansClusterer.cluster() as `faultPercent`.
  - Shift split: each shift window's real historical share of fault
                 OCCURRENCES (faults.created_at via
                 DataExtractor.get_fault_shift_distribution) — NOT job
                 completions (jobs.completed_at), which is a different
                 signal Stage 1 built for a different purpose (Technician
                 Suggestion's capacity denominator, see below). Chosen
                 explicitly over reusing the job-completion data or an
                 even 1/3 split, since neither existing signal answers
                 "when do faults actually occur" on its own.

Advisory only, per SRS 5.6.8: this module produces suggested quantities: it
does not write anything to any BOD dispatch screen — that pre-population is
a Stage 3 (Admin UI) concern, out of scope here.
"""

import logging
import math
from datetime import datetime
from typing import Optional

import pandas as pd

from config import Config

logger = logging.getLogger('slt_ai.resource_planner')

# SRS 5.6.8 Time Window Prediction — the only three shift windows it defines.
# Matches the CASE expressions in DataExtractor.get_fault_shift_distribution
# and get_technician_shift_performance exactly; faults/completions outside
# these hours (22:00-06:00) are excluded upstream in both queries.
SHIFT_WINDOWS = ['MORNING', 'AFTERNOON', 'EVENING']


class ResourcePlanner:
    """
    Stateless combiner over an already-constructed ProphetForecaster,
    KMeansClusterer, and DataExtractor. Holds references, not its own model.
    """

    def __init__(self, forecaster, clusterer, extractor):
        self.forecaster = forecaster
        self.clusterer = clusterer
        self.extractor = extractor

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────────────────────

    def generate_plan(
        self,
        raw_fault_df: pd.DataFrame,
        gps_df: pd.DataFrame,
        horizon_days: int = 7,
    ) -> dict:
        """
        Main entry point — GET /api/ai/resource-plan.

        Args:
            raw_fault_df: [ds, y] daily fault counts (same shape ProphetForecaster
                          .forecast() and /api/ai/predictions already use).
            gps_df:       fault GPS points for KMeansClusterer.cluster() (same
                          shape /api/ai/clusters already uses).
            horizon_days: Days ahead to plan for (7-90, same range as
                          /api/ai/predictions' horizon param).

        Returns (insufficient data — SRS 5.6.2/5.6.8 shared 6-month rule,
        via ProphetForecaster's own has_sufficient_history() check):
            {insufficientData: true, historyDays, requiredDays,
             hotspots: [], materialShortfalls: []}

        Returns (normal):
            {
              insufficientData: false,
              horizonDays: int,
              hotspots: [{
                date, shift, zoneId, zoneName,
                predictedFaultCount,
                suggestedTechnicians, suggestedVehicles,
                materials: [{materialId, materialName, suggestedQuantity, unit}],
              }, ...],  # sorted by predictedFaultCount descending
              materialShortfalls: [{
                materialId, materialName, currentStock,
                totalSuggestedQuantity, insufficient, unit
              }],
              shiftDistributionSource: 'historical' | 'even_split_fallback',
              generatedAt: ISO8601 str
            }
        """
        forecast_result = self.forecaster.forecast(raw_fault_df, horizon=horizon_days)
        if forecast_result.get('insufficientData'):
            return {
                'insufficientData': True,
                'historyDays':      forecast_result.get('historyDays'),
                'requiredDays':     forecast_result.get('requiredDays'),
                'hotspots':          [],
                'materialShortfalls': [],
            }

        cluster_result = self.clusterer.cluster(gps_df, n_clusters=Config.KMEANS_N_CLUSTERS)
        zones = cluster_result.get('clusters', [])
        total_faults = cluster_result.get('totalFaults', 0)

        shift_pct, shift_distribution_source = self._normalise_shift_distribution(
            self.extractor.get_fault_shift_distribution(days_back=180)
        )
        avg_jobs_per_tech_shift = self._compute_avg_jobs_per_tech_shift(
            self.extractor.get_technician_shift_performance(days_back=180)
        )
        material_rates = self._compute_material_rates(
            self.extractor.get_material_usage_with_location(days_back=365),
            zones, total_faults,
        )
        stock_by_material = self._index_stock(self.extractor.get_material_stock())

        hotspots = self._build_hotspots(
            forecast_result.get('forecast', []),
            zones, shift_pct, avg_jobs_per_tech_shift, material_rates,
        )
        hotspots.sort(key=lambda h: h['predictedFaultCount'], reverse=True)

        material_shortfalls = self._compute_shortfalls(hotspots, stock_by_material)

        return {
            'insufficientData': False,
            'horizonDays':       horizon_days,
            'hotspots':          hotspots,
            'materialShortfalls': material_shortfalls,
            'shiftDistributionSource': shift_distribution_source,
            'generatedAt':       datetime.utcnow().isoformat() + 'Z',
        }

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE — DISTRIBUTION RATIOS
    # ─────────────────────────────────────────────────────────────────────────

    def _normalise_shift_distribution(self, shift_df: Optional[pd.DataFrame]) -> tuple:
        """
        (pct, source) — pct is {shift: fraction_of_faults}, summing to 1.0
        across SHIFT_WINDOWS; source is 'historical' or 'even_split_fallback'.

        Falls back to an even 1/3 split only when there's literally no
        historical shift data at all (e.g. a brand-new deployment) — this
        is a graceful degradation of the chosen method for a missing-data
        edge case, not a silent substitution of the method itself. `source`
        is what makes that degradation visible to a caller instead of
        indistinguishable from a genuine historical result (the transparency
        gap this was found to have, alongside the already-existing
        `insufficientData` pattern for the forecast half of the same
        response — see generate_plan's `shiftDistributionSource`).
        """
        if shift_df is None or shift_df.empty:
            return {s: 1 / 3 for s in SHIFT_WINDOWS}, 'even_split_fallback'
        total = shift_df['fault_count'].sum()
        if total <= 0:
            return {s: 1 / 3 for s in SHIFT_WINDOWS}, 'even_split_fallback'
        pct = {row['shift']: row['fault_count'] / total for _, row in shift_df.iterrows()}
        for s in SHIFT_WINDOWS:
            pct.setdefault(s, 0.0)
        return pct, 'historical'

    def _compute_avg_jobs_per_tech_shift(self, tech_shift_df: Optional[pd.DataFrame]) -> dict:
        """
        {shift: avg_jobs_per_technician_per_day_worked_that_shift}, pooled
        across all technicians (sum(jobs_completed) / sum(days_covered) per
        shift, not an unweighted mean-of-means) — robust to one technician
        having far more/fewer working days on record than another. `None`
        for a shift with zero historical technician-days on record, rather
        than fabricating a rate.
        """
        result = {s: None for s in SHIFT_WINDOWS}
        if tech_shift_df is None or tech_shift_df.empty:
            return result
        grouped = tech_shift_df.groupby('shift').agg(
            total_jobs=('jobs_completed', 'sum'),
            total_days=('days_covered', 'sum'),
        )
        for shift, row in grouped.iterrows():
            if shift in result and row['total_days'] > 0:
                result[shift] = float(row['total_jobs']) / float(row['total_days'])
        return result

    def _compute_material_rates(
        self,
        usage_df: Optional[pd.DataFrame],
        zones: list,
        total_faults: int,
    ) -> dict:
        """
        {clusterId: {materialId: {'name': str, 'unit': str|None, 'avgQtyPerFault': float}}}

        Historical average quantity used per fault, per material, per zone.
        Denominator (fault count per zone) comes from the already-computed
        `faultPercent` in `zones` (cluster_result['clusters']) rather than a
        separate independent recount, so it stays consistent with the same
        numbers the /api/ai/clusters overview reports.

        Zone assignment is done here (not by DataExtractor — see
        get_material_usage_with_location's docstring) via
        KMeansClusterer.assign_zones(), which requires cluster() to have
        already run in this same request (generate_plan() guarantees that
        ordering).
        """
        rates: dict = {}
        if usage_df is None or usage_df.empty or not zones or total_faults <= 0:
            return rates

        usage_df = usage_df.dropna(subset=['latitude', 'longitude']).copy()
        if usage_df.empty:
            return rates

        usage_df['cluster_id'] = self.clusterer.assign_zones(usage_df)

        zone_fault_counts = {
            int(z['clusterId']): total_faults * (z.get('faultPercent', 0) / 100.0)
            for z in zones
        }

        grouped = usage_df.groupby(['cluster_id', 'material_id']).agg(
            material_name=('material_name', 'first'),
            total_qty=('quantity_used', 'sum'),
        ).reset_index()

        for _, row in grouped.iterrows():
            zid = int(row['cluster_id'])
            fault_count = zone_fault_counts.get(zid, 0)
            if fault_count <= 0:
                continue
            rates.setdefault(zid, {})[int(row['material_id'])] = {
                'name':          row['material_name'],
                'avgQtyPerFault': float(row['total_qty']) / fault_count,
            }
        return rates

    def _index_stock(self, stock_df: Optional[pd.DataFrame]) -> dict:
        """{materialId: {'name', 'currentStock', 'unit'}}"""
        if stock_df is None or stock_df.empty:
            return {}
        return {
            int(row['material_id']): {
                'name':         row['material_name'],
                'currentStock': float(row['current_stock']) if row['current_stock'] is not None else 0.0,
                'unit':         row.get('unit'),
            }
            for _, row in stock_df.iterrows()
        }

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE — HOTSPOT / SUGGESTION BUILDING
    # ─────────────────────────────────────────────────────────────────────────

    def _build_hotspots(
        self,
        forecast_days: list,
        zones: list,
        shift_pct: dict,
        avg_jobs_per_tech_shift: dict,
        material_rates: dict,
    ) -> list:
        hotspots = []
        for day in forecast_days:
            day_total = max(float(day.get('yhat', 0) or 0), 0.0)
            date = day.get('ds')

            for zone in zones:
                zone_id   = int(zone['clusterId'])
                zone_name = zone.get('regionName', f'Zone {zone_id}')
                zone_pct  = zone.get('faultPercent', 0) / 100.0

                for shift in SHIFT_WINDOWS:
                    predicted_count = day_total * zone_pct * shift_pct.get(shift, 0)

                    avg_jobs = avg_jobs_per_tech_shift.get(shift)
                    suggested_technicians = (
                        math.ceil(predicted_count / avg_jobs)
                        if avg_jobs and avg_jobs > 0
                        else None
                    )
                    # SRS 5.6.8 — Vehicle Suggestion matches Technician count 1:1.
                    suggested_vehicles = suggested_technicians

                    materials = []
                    for material_id, rate in material_rates.get(zone_id, {}).items():
                        materials.append({
                            'materialId':        material_id,
                            'materialName':      rate['name'],
                            'suggestedQuantity': round(rate['avgQtyPerFault'] * predicted_count, 2),
                        })

                    hotspots.append({
                        'date':                 date,
                        'shift':                shift,
                        'zoneId':               zone_id,
                        'zoneName':             zone_name,
                        'predictedFaultCount':  round(predicted_count, 2),
                        'suggestedTechnicians': suggested_technicians,
                        'suggestedVehicles':    suggested_vehicles,
                        'materials':            materials,
                    })
        return hotspots

    def _compute_shortfalls(self, hotspots: list, stock_by_material: dict) -> list:
        """
        Total suggested quantity per material across every hotspot in the
        horizon, vs. current stock — SRS 5.6.8: "flags materials where
        current stock is insufficient to cover the predicted demand."
        """
        totals: dict = {}
        for h in hotspots:
            for m in h['materials']:
                mid = m['materialId']
                totals[mid] = totals.get(mid, 0.0) + m['suggestedQuantity']

        shortfalls = []
        for mid, total_qty in totals.items():
            stock_info = stock_by_material.get(mid, {})
            current_stock = stock_info.get('currentStock', 0.0)
            shortfalls.append({
                'materialId':             mid,
                'materialName':           stock_info.get('name', f'Material {mid}'),
                'currentStock':           current_stock,
                'totalSuggestedQuantity': round(total_qty, 2),
                'unit':                   stock_info.get('unit'),
                'insufficient':           current_stock < total_qty,
            })
        shortfalls.sort(key=lambda s: s['totalSuggestedQuantity'], reverse=True)
        return shortfalls
