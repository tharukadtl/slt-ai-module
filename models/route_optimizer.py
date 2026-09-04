"""
models/route_optimizer.py — DijkstraRouter (FR-29, SRS 5.6.6)
=================================================================
Technician-to-fault shortest-path navigation.

FR-28 ("find nearest available technician to a reported fault") was
removed from the spec in v1.5 and its implementation deleted. This
module now implements the unrelated FR-29 capability instead: given
ONE technician's live location and their assigned fault's location,
compute the shortest path between them.

This is Stage 1 — the AI/ML backend contract only (POST
/api/ai/shortest-path in app.py). Mobile-side rendering is Stage 2.

Road-network graph data (an OSM way graph, an OSRM/GraphHopper/Valhalla
instance, a cached graph file, etc.) does not exist anywhere in this
codebase: requirements.txt has no osmnx/networkx/routing-engine client,
and no cached graph data is checked in. SRS 5.6.6 explicitly allows for
this — "road graph (Haversine fallback)" — so every request is
currently served via the fallback path.

The router still runs a real Dijkstra shortest-path search over an
explicit graph (see `_dijkstra`) rather than special-casing the 2-node
case, for two reasons:
  - SRS 5.6.6 names Dijkstra's algorithm specifically, not just "compute
    a distance"
  - if real road-graph data becomes available later, feeding a graph
    with intermediate nodes (intersections / road segments) into the
    same `route()` / `_dijkstra()` produces a real multi-waypoint route
    with no change to the endpoint contract — only `build_graph()`
    would need to change.
"""

import heapq
import logging
from typing import Dict, List, Optional, Tuple

from config import Config
from data.feature_engineer import haversine_km

logger = logging.getLogger('slt_ai.router')


class DijkstraRouter:
    """
    Computes Technician -> fault shortest paths (FR-29 / SRS 5.6.6).

    Always runs in Haversine-fallback mode today (no road-network graph
    data source is wired up) but is structured so a real routed graph
    can be swapped into `build_graph()` later without callers changing.
    """

    def __init__(self, avg_speed_kmh: float = None):
        self.avg_speed_kmh = avg_speed_kmh or Config.ROUTE_AVG_SPEED_KMH
        logger.info(
            "DijkstraRouter initialised (Haversine-fallback mode — "
            "no road-network graph data source configured)"
        )

    # ─────────────────────────────────────────────────────────────────────
    # GRAPH CONSTRUCTION
    # ─────────────────────────────────────────────────────────────────────

    def build_graph(
        self,
        start_lat: float, start_lng: float,
        end_lat: float, end_lng: float,
    ) -> Dict:
        """
        Build the graph `_dijkstra` runs over.

        No road-network data source is available, so this is a 2-node
        graph (start, end) with a single Haversine-weighted edge — the
        geometric shortest path. A future real-routing implementation
        replaces only this method (adding intersection nodes between
        start/end with edges weighted by real road-segment distances);
        `route()` and `_dijkstra()` do not need to change.
        """
        dist = haversine_km(start_lat, start_lng, end_lat, end_lng)
        nodes = {
            'start': {'lat': start_lat, 'lng': start_lng},
            'end':   {'lat': end_lat,   'lng': end_lng},
        }
        edges = {
            'start': [('end', dist)],
            'end':   [('start', dist)],
        }
        return {'nodes': nodes, 'edges': edges, 'routed': False}

    # ─────────────────────────────────────────────────────────────────────
    # DIJKSTRA
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _dijkstra(
        nodes: Dict, edges: Dict, source: str, target: str
    ) -> Tuple[List[str], float]:
        """
        Standard Dijkstra shortest path over an adjacency structure
        {node_id: [(neighbour_id, weight_km), ...]}.

        Returns (ordered_node_id_path, total_distance_km). Path is empty
        and distance is inf if target is unreachable from source.
        """
        dist = {n: float('inf') for n in nodes}
        prev: Dict[str, Optional[str]] = {n: None for n in nodes}
        dist[source] = 0.0
        visited = set()
        heap = [(0.0, source)]

        while heap:
            d, u = heapq.heappop(heap)
            if u in visited:
                continue
            visited.add(u)
            if u == target:
                break
            for v, w in edges.get(u, []):
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(heap, (nd, v))

        if dist[target] == float('inf'):
            return [], float('inf')

        path = []
        node = target
        while node is not None:
            path.append(node)
            node = prev[node]
        path.reverse()
        return path, dist[target]

    # ─────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────────────────

    def route(
        self,
        current_lat: float, current_lng: float,
        fault_lat: float, fault_lng: float,
    ) -> Dict:
        """
        Compute the shortest path from a Technician's current live
        location to their assigned fault's location (FR-29 / SRS 5.6.6).

        Returns:
            {
              'waypoints':   [{'lat', 'lng'}, ...] ordered start -> end,
              'distanceKm':  float,
              'etaMinutes':  int,
              'routed':      bool — True if a real road-network path was
                             used, False if this is the Haversine
                             straight-line fallback (always False today),
              'algorithm':   str — human-readable description,
              'avgSpeedKmh': float — the speed assumption behind etaMinutes,
            }
        """
        graph = self.build_graph(current_lat, current_lng, fault_lat, fault_lng)
        path_ids, distance_km = self._dijkstra(graph['nodes'], graph['edges'], 'start', 'end')

        waypoints = [
            {'lat': graph['nodes'][nid]['lat'], 'lng': graph['nodes'][nid]['lng']}
            for nid in path_ids
        ]

        eta_minutes = round((distance_km / self.avg_speed_kmh) * 60) if self.avg_speed_kmh else None

        return {
            'waypoints':   waypoints,
            'distanceKm':  round(distance_km, 3),
            'etaMinutes':  eta_minutes,
            'routed':      graph['routed'],
            'algorithm': (
                'dijkstra (road-network graph)' if graph['routed']
                else 'dijkstra (haversine-fallback, straight-line)'
            ),
            'avgSpeedKmh': self.avg_speed_kmh,
        }
