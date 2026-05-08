"""
models/route_optimizer.py — Dijkstra + Haversine Route Optimiser
=================================================================
Finds the nearest available technicians to a fault location using
Dijkstra's shortest-path algorithm on a GPS-coordinate graph,
with Haversine formula for edge weights (great-circle distances).

SRS requirement:
  - Find nearest available technician to a reported fault
  - Target: 15–20% reduction in technician travel time vs manual assignment
  - Operates on real-time technician GPS positions from MySQL
  - Falls back to synthetic technician positions in demo mode

Algorithm:
  - Build a fully-connected weighted graph from technician GPS positions
  - Edge weights = Haversine distance (km) between every pair of nodes
  - Run Dijkstra from the fault location node
  - Return top-k nearest technicians ranked by distance
  - Estimate arrival time using 40 km/h average SL road speed

Usage:
    from models.route_optimizer import DijkstraRouter
    dr = DijkstraRouter()
    result = dr.find_nearest(technicians_df, fault_lat=6.93, fault_lng=79.86)
    # result: { technicians: [{rank, name, distance, eta, ...}], algorithm }
"""

import heapq
import logging
import math
from datetime import datetime
from typing import List, Optional

import pandas as pd

from config import Config
from data.feature_engineer import FeatureEngineer, haversine_km, estimate_travel_minutes

logger = logging.getLogger('slt_ai.router')

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Average road speed (km/h) — conservative for Sri Lanka urban + rural mix
AVG_SPEED_URBAN_KMH  = 30.0
AVG_SPEED_RURAL_KMH  = 50.0
URBAN_RADIUS_KM      = 15.0   # within 15 km of Colombo → use urban speed

# Colombo coordinates for urban/rural classification
COLOMBO_LAT = 6.9271
COLOMBO_LNG = 79.8612

# Status weights — factor applied to distance for priority scoring
# Available technician is preferred over one already on a job
STATUS_WEIGHT = {
    'AVAILABLE': 1.0,
    'ACCEPTED':  1.0,
    'PAUSED':    1.3,    # slightly less preferred
    'TRAVELLING': 1.5,
    'IN_PROGRESS': 2.0,  # discourage assigning another job
    'OFFLINE':   99.0,   # effectively unreachable
    'BREAK':     1.8,
}

# Maximum search radius (km) — beyond this, technician is considered too far
MAX_SEARCH_RADIUS_KM = Config.ROUTE_SEARCH_RADIUS_KM


class DijkstraRouter:
    """
    Implements Dijkstra's shortest-path algorithm on a geographic graph
    of technician GPS positions to find the optimal assignment for a fault.
    """

    def __init__(self):
        self._fe = FeatureEngineer()
        logger.info("DijkstraRouter initialised")

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────────────────────

    def find_nearest(
        self,
        technicians_df: pd.DataFrame,
        fault_lat: float,
        fault_lng: float,
        limit: int = 5,
        available_only: bool = False,
    ) -> dict:
        """
        Main entry point: find nearest technicians to a fault location.

        1. Clean technician locations
        2. Build weighted GPS graph
        3. Run Dijkstra from fault node
        4. Rank results and add metadata

        Args:
            technicians_df: DataFrame with technician GPS and status data.
            fault_lat:      Fault location latitude.
            fault_lng:      Fault location longitude.
            limit:          Max technicians to return (default 5).
            available_only: Only return AVAILABLE / ACCEPTED status (default False).

        Returns:
            {
              technicians: [{
                rank, technicianId, technicianName, phone, branchName,
                latitude, longitude, distanceKm, weightedDistanceKm,
                estimatedArrivalMinutes, status, currentJobId,
                isAvailable, avatarInitial
              }],
              algorithm: "dijkstra+haversine",
              totalCandidates: int,
              searchRadiusKm:  float,
            }
        """
        # ── 1. Clean technician locations ─────────────────────────────────────
        from data.data_cleaner import DataCleaner
        cleaner = DataCleaner()
        tech_df = cleaner.clean_technician_locations(technicians_df)

        if tech_df is None or tech_df.empty:
            logger.warning("No valid technician locations — returning empty result")
            return self._empty_result()

        # ── 2. Optional: filter by availability ───────────────────────────────
        if available_only:
            tech_df = tech_df[
                tech_df['status'].isin(Config.TECH_AVAILABLE_STATUSES)
            ]
            if tech_df.empty:
                logger.info("No available technicians found — returning all with weights")
                tech_df = cleaner.clean_technician_locations(technicians_df)

        # ── 3. Filter by search radius (performance + relevance) ──────────────
        tech_df = self._filter_by_radius(tech_df, fault_lat, fault_lng)

        if tech_df is None or tech_df.empty:
            logger.warning(f"No technicians within {MAX_SEARCH_RADIUS_KM} km radius")
            # Expand search and try again with full list
            tech_df = cleaner.clean_technician_locations(technicians_df)
            if tech_df is None or tech_df.empty:
                return self._empty_result()

        # ── 4. Build graph ────────────────────────────────────────────────────
        graph = self._fe.build_distance_graph(tech_df, fault_lat, fault_lng)

        # ── 5. Dijkstra ───────────────────────────────────────────────────────
        distances = self._dijkstra(graph, source='fault_0')

        # ── 6. Build ranked results ───────────────────────────────────────────
        results = self._build_results(
            distances, graph, fault_lat, fault_lng, limit
        )

        return {
            'technicians':      results,
            'algorithm':        'dijkstra+haversine',
            'totalCandidates':  len(tech_df),
            'searchRadiusKm':   MAX_SEARCH_RADIUS_KM,
            'computedAt':       datetime.utcnow().isoformat() + 'Z',
        }

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE — DIJKSTRA ALGORITHM
    # ─────────────────────────────────────────────────────────────────────────

    def _dijkstra(self, graph: dict, source: str) -> dict:
        """
        Standard Dijkstra's algorithm using a binary min-heap (heapq).

        Args:
            graph:  {'nodes': {...}, 'edges': {node_id: [(neighbour, weight)]}}
            source: Source node ID (the fault location: 'fault_0')

        Returns:
            dict: {node_id: shortest_distance_from_source}
        """
        nodes = graph['nodes']
        edges = graph['edges']

        # Priority queue: (distance, node_id)
        pq   = [(0.0, source)]
        dist = {nid: math.inf for nid in nodes}
        dist[source] = 0.0
        visited = set()

        while pq:
            current_dist, current_node = heapq.heappop(pq)

            if current_node in visited:
                continue
            visited.add(current_node)

            # Relax edges
            for neighbour, weight in edges.get(current_node, []):
                if neighbour in visited:
                    continue

                # Apply status weight to penalise unavailable technicians
                node_data = nodes.get(neighbour, {})
                status    = node_data.get('status', 'AVAILABLE')
                sw        = STATUS_WEIGHT.get(status, 1.0)
                effective_weight = weight * sw

                new_dist = current_dist + effective_weight
                if new_dist < dist[neighbour]:
                    dist[neighbour] = new_dist
                    heapq.heappush(pq, (new_dist, neighbour))

        return dist

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE — RESULT BUILDING
    # ─────────────────────────────────────────────────────────────────────────

    def _build_results(
        self,
        distances:  dict,
        graph:      dict,
        fault_lat:  float,
        fault_lng:  float,
        limit:      int
    ) -> List[dict]:
        """
        Convert Dijkstra distance dict to ranked technician list.

        For each technician node:
        - Compute true Haversine distance (raw, without status penalty)
        - Estimate arrival time at appropriate speed
        - Add all technician metadata from graph nodes
        """
        nodes = graph['nodes']
        results = []

        for node_id, weighted_dist in distances.items():
            if node_id == 'fault_0':
                continue
            if math.isinf(weighted_dist):
                continue

            node_data = nodes.get(node_id, {})
            if node_data.get('type') != 'technician':
                continue

            t_lat = float(node_data['lat'])
            t_lng = float(node_data['lng'])

            # True (unweighted) geographic distance
            true_dist_km = haversine_km(fault_lat, fault_lng, t_lat, t_lng)

            # ETA — use urban or rural speed based on fault distance from Colombo
            fault_dist_from_colombo = haversine_km(
                fault_lat, fault_lng, COLOMBO_LAT, COLOMBO_LNG
            )
            speed = (
                AVG_SPEED_URBAN_KMH
                if fault_dist_from_colombo <= URBAN_RADIUS_KM
                else AVG_SPEED_RURAL_KMH
            )
            eta_minutes = estimate_travel_minutes(true_dist_km, avg_speed_kmh=speed)

            status      = node_data.get('status', 'UNKNOWN')
            is_available = status in Config.TECH_AVAILABLE_STATUSES

            results.append({
                'technicianId':           int(node_data.get('technician_id', 0)),
                'technicianName':         str(node_data.get('full_name', 'Unknown')),
                'phone':                  str(node_data.get('phone', '')),
                'branchName':             str(node_data.get('branch_name', '')),
                'latitude':               round(t_lat, 6),
                'longitude':              round(t_lng, 6),
                'distanceKm':             round(true_dist_km, 2),
                'weightedDistanceKm':     round(weighted_dist, 2),
                'estimatedArrivalMinutes': eta_minutes,
                'status':                 status,
                'isAvailable':            is_available,
                'currentJobId':           node_data.get('current_job_id'),
                'avatarInitial':          str(node_data.get('full_name', 'T'))[0].upper(),
            })

        # Sort by weighted distance (Dijkstra result = status-aware ranking)
        results.sort(key=lambda r: r['weightedDistanceKm'])

        # Add rank and return top-k
        for i, r in enumerate(results[:limit]):
            r['rank'] = i + 1

        return results[:limit]

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE — HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _filter_by_radius(
        self,
        tech_df:   pd.DataFrame,
        fault_lat: float,
        fault_lng: float,
    ) -> Optional[pd.DataFrame]:
        """
        Return only technicians within MAX_SEARCH_RADIUS_KM of the fault.
        Computes Haversine distance for each row.
        """
        if tech_df is None or tech_df.empty:
            return None

        tech_df = tech_df.copy()
        tech_df['_dist_km'] = tech_df.apply(
            lambda row: haversine_km(
                fault_lat, fault_lng,
                float(row['latitude']), float(row['longitude'])
            ),
            axis=1
        )
        filtered = tech_df[tech_df['_dist_km'] <= MAX_SEARCH_RADIUS_KM].copy()
        filtered = filtered.drop(columns=['_dist_km'])
        return filtered if not filtered.empty else None

    def _empty_result(self) -> dict:
        """Return a safe empty result when no technicians are found."""
        return {
            'technicians':     [],
            'algorithm':       'dijkstra+haversine',
            'totalCandidates': 0,
            'searchRadiusKm':  MAX_SEARCH_RADIUS_KM,
            'computedAt':      datetime.utcnow().isoformat() + 'Z',
            'note':            'No technicians found within search radius',
        }

    # ─────────────────────────────────────────────────────────────────────────
    # BATCH / MULTI-FAULT ASSIGNMENT
    # ─────────────────────────────────────────────────────────────────────────

    def batch_assign(
        self,
        technicians_df: pd.DataFrame,
        faults: List[dict]
    ) -> List[dict]:
        """
        Greedy batch assignment — assign nearest available technician to
        each fault in priority order, marking techs as unavailable once assigned.

        Args:
            technicians_df: Technician locations DataFrame.
            faults: List of { fault_id, lat, lng, priority } dicts,
                    sorted by priority (HIGH first).

        Returns:
            List of { fault_id, technician_id, distance_km, eta_minutes }
        """
        from data.data_cleaner import DataCleaner

        cleaner  = DataCleaner()
        tech_df  = cleaner.clean_technician_locations(technicians_df)

        if tech_df is None or tech_df.empty:
            return [{'fault_id': f['fault_id'], 'error': 'No technicians available'}
                    for f in faults]

        assigned_ids = set()
        assignments  = []

        for fault in faults:
            # Mark already-assigned techs as unavailable
            available = tech_df[~tech_df['technician_id'].isin(assigned_ids)]
            if available.empty:
                assignments.append({
                    'fault_id': fault['fault_id'],
                    'error':    'No unassigned technicians available',
                })
                continue

            result = self.find_nearest(
                available,
                fault_lat=fault['lat'],
                fault_lng=fault['lng'],
                limit=1,
            )

            techs = result.get('technicians', [])
            if techs:
                best = techs[0]
                assignments.append({
                    'fault_id':                fault['fault_id'],
                    'technician_id':           best['technicianId'],
                    'technician_name':         best['technicianName'],
                    'distance_km':             best['distanceKm'],
                    'estimated_arrival_minutes': best['estimatedArrivalMinutes'],
                })
                assigned_ids.add(best['technicianId'])
            else:
                assignments.append({
                    'fault_id': fault['fault_id'],
                    'error':    'No technician found within search radius',
                })

        logger.info(
            f"Batch assignment: {len(faults)} faults, "
            f"{sum(1 for a in assignments if 'technician_id' in a)} assigned"
        )
        return assignments
