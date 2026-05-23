"""Abstract base class for all parking maps.

Any new parking-lot layout only needs to inherit ``ParkingMap`` and implement
the five abstract properties / methods listed below.  The rest of the system
(planner, collision, heuristics, visualization) works through this interface
and requires **zero** modification when a new map is added.
"""

import math
import heapq
from abc import ABC, abstractmethod

import numpy as np

try:
    from scipy.ndimage import distance_transform_edt, maximum_filter
except ImportError:
    distance_transform_edt = None
    maximum_filter = None


class ParkingMap(ABC):
    """Abstract interface that every parking-lot map must implement."""

    # ------------------------------------------------------------------
    # Required abstract interface
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def obstacles(self) -> list:
        """Return list of (x, y) obstacle sample points."""

    @property
    @abstractmethod
    def obs_arr(self) -> np.ndarray:
        """Return Nx2 numpy array of obstacle sample points."""

    @property
    @abstractmethod
    def map_bounds(self) -> dict:
        """Return dict with keys: x_min, x_max, y_min, y_max, width, height.

        Used by Dijkstra grid sizing and visualization bounding box.
        """

    @abstractmethod
    def get_directional_rules(self) -> dict:
        """Return directional rules config dict (may be empty for no rules)."""

    @abstractmethod
    def get_visualization_elements(self) -> dict:
        """Return dict describing how to draw this map.

        Expected keys (all optional — draw nothing if missing):
          - ``walls``:  list of {'points': [(x1,y1),(x2,y2)], 'style': 'wall'|'env'}
          - ``slot_lines``: list of {'x_range': (x0,x1), 'y_values': [y,...], 'style': 'env'}
          - ``aisle_guides``: list of {'x': float, 'y_range': (y0,y1), 'style': 'open'}
          - ``labels``: list of {'text': str, 'x': float, 'y': float, 'ha': str, 'va': str}
          - ``entrance``: {'x_range': (x0,x1), 'y': float}  (for the entrance label)
        """

    @abstractmethod
    def get_start_state_params(self) -> dict:
        """Return default start state parameters.

        Expected keys:
          - ``x``: float
          - ``y``: float
          - ``theta``: float  (radians)
        """

    @abstractmethod
    def get_parking_scenarios_config(self) -> dict:
        """Return configuration for generating parking scenarios.

        Expected keys:
          - ``spots``:  list of dicts, each with:
              - ``name``: str (e.g. 'Q1_upper_left')
              - ``center_x``: float
              - ``center_y_options``: list[float]
              - ``default_y_index``: int  (1-based)
              - ``outward_theta``: float  (radians)
          - ``center_wall_x``: float  (used to determine outward_theta if needed)
        """

    # ------------------------------------------------------------------
    # Concrete shared methods (common to all maps)
    # ------------------------------------------------------------------

    def __init__(self):
        self.dijkstra_grid = None
        self.grid_res = 1.0
        self._directional_rules = None
        self._skeleton_dist_grid = None
        self._skeleton_meta = None
        self._skeleton_warned = False

    def get_do(self, x, y):
        """Distance to the nearest obstacle from point (x, y)."""
        dists = np.hypot(self.obs_arr[:, 0] - x, self.obs_arr[:, 1] - y)
        return np.min(dists)

    def update_directional_rules(self, rules_dict):
        """Override directional rules at runtime."""
        self._directional_rules = rules_dict

    def precompute_2d_dijkstra(self, goal, directional=False):
        """BFS / Dijkstra on a 2-D grid with optional directional penalty."""
        bounds = self.map_bounds
        if directional:
            print('>> 正在預先計算「具方向性」的 2D Dijkstra 障礙物感知地圖...')
        else:
            print('>> 正在預先計算 2D Dijkstra 障礙物感知地圖...')

        w = int((bounds['x_max'] + 2.0) / self.grid_res)
        h = int((bounds['y_max'] + 5.0) / self.grid_res)
        self.dijkstra_grid = np.full((w + 1, h + 1), float('inf'))

        gx = int(goal.x / self.grid_res)
        gy = int(goal.y / self.grid_res)
        if not (0 <= gx <= w and 0 <= gy <= h):
            return
        self.dijkstra_grid[gx, gy] = 0.0

        queue = [(0.0, gx, gy)]
        motions = [(1, 0), (0, 1), (-1, 0), (0, -1),
                   (1, 1), (-1, 1), (1, -1), (-1, -1)]

        # 讀取走廊範圍用於方向性懲罰
        aisle_ranges = []
        if directional:
            rules = self.get_directional_rules()
            for aisle in rules.get('aisles', []):
                aisle_ranges.append({
                    'x_range': aisle['x_range'],
                    'forbidden_dy_sign': aisle['forbidden_dy_sign'],
                })

        while queue:
            cost, cx, cy = heapq.heappop(queue)
            for dx, dy in motions:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx <= w and 0 <= ny <= h:
                    real_x = nx * self.grid_res
                    real_y = ny * self.grid_res
                    if self.get_do(real_x, real_y) < 1.0:
                        continue

                    step_cost = math.hypot(dx, dy) * self.grid_res
                    penalty_multiplier = 1.0

                    if directional:
                        car_dy = -dy
                        for ar in aisle_ranges:
                            x_min, x_max = ar['x_range']
                            if x_min <= real_x <= x_max:
                                fds = ar['forbidden_dy_sign']
                                # fds=-1 → upward flow → car going down is bad
                                # fds= 1 → downward flow → car going up is bad
                                if (fds == -1 and car_dy < 0) or (fds == 1 and car_dy > 0):
                                    penalty_multiplier = 3.0
                                break

                    n_cost = cost + (step_cost * penalty_multiplier)
                    if n_cost < self.dijkstra_grid[nx, ny]:
                        self.dijkstra_grid[nx, ny] = n_cost
                        heapq.heappush(queue, (n_cost, nx, ny))

    def invalidate_skeleton_cache(self):
        """Invalidate cached skeleton distance fields (for dynamic obstacles)."""
        self._skeleton_dist_grid = None
        self._skeleton_meta = None

    def _build_occupancy_grid(self):
        bounds = self.map_bounds
        grid_res = float(self.grid_res)
        x_min = float(bounds['x_min'])
        y_min = float(bounds['y_min'])
        x_max = float(bounds['x_max'])
        y_max = float(bounds['y_max'])

        x_w = int(round((x_max - x_min) / grid_res)) + 1
        y_w = int(round((y_max - y_min) / grid_res)) + 1
        obs_grid = np.zeros((x_w, y_w), dtype=bool)

        if self.obs_arr.size > 0:
            ix = np.round((self.obs_arr[:, 0] - x_min) / grid_res).astype(int)
            iy = np.round((self.obs_arr[:, 1] - y_min) / grid_res).astype(int)
            mask = (ix >= 0) & (ix < x_w) & (iy >= 0) & (iy < y_w)
            obs_grid[ix[mask], iy[mask]] = True

        meta = {
            'x_min': x_min,
            'y_min': y_min,
            'x_w': x_w,
            'y_w': y_w,
            'grid_res': grid_res,
        }
        return obs_grid, meta

    def _compute_skeleton_distance_field(self):
        if distance_transform_edt is None or maximum_filter is None:
            if not self._skeleton_warned:
                print('>> [Skeleton] scipy.ndimage not available, fallback to aisle center.')
                self._skeleton_warned = True
            return False

        obs_grid, meta = self._build_occupancy_grid()
        if obs_grid.size == 0:
            return False

        free_grid = ~obs_grid
        d_obs = distance_transform_edt(free_grid) * meta['grid_res']

        local_max = maximum_filter(d_obs, size=5) == d_obs
        local_max[d_obs <= 0.0] = False
        if not np.any(local_max):
            return False

        d_skel = distance_transform_edt(~local_max) * meta['grid_res']
        self._skeleton_dist_grid = d_skel
        self._skeleton_meta = meta
        return True

    def get_skeleton_distance(self, x, y):
        """Return distance to the skeleton centerline, if available."""
        if self._skeleton_dist_grid is None:
            if not self._compute_skeleton_distance_field():
                return None

        meta = self._skeleton_meta
        if meta is None:
            return None

        ix = int(round((x - meta['x_min']) / meta['grid_res']))
        iy = int(round((y - meta['y_min']) / meta['grid_res']))
        if not (0 <= ix < meta['x_w'] and 0 <= iy < meta['y_w']):
            return None

        return float(self._skeleton_dist_grid[ix, iy])
