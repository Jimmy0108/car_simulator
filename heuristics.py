import math
import os
import time
import numpy as np
import reeds_shepp as rs

from collision import is_collision_free, rs_collision_free
from config import PARAMS
from models import State


class NonHolonomicHeuristicTable:
    """Lookup table for obstacle-free non-holonomic cost (RS path length)."""

    def __init__(self, xy_res=None, xy_extent=None, theta_bins=None, turning_radius=None):
        self.xy_res = float(xy_res if xy_res is not None else PARAMS['h_table_xy_res'])
        self.xy_extent = float(xy_extent if xy_extent is not None else PARAMS['h_table_xy_extent'])
        self.theta_bins = int(theta_bins if theta_bins is not None else PARAMS['h_table_theta_bins'])
        self.turning_radius = float(turning_radius if turning_radius is not None else PARAMS['turning_radius'])

        self.x_cells = int(round((2.0 * self.xy_extent) / self.xy_res)) + 1
        self.y_cells = int(round((2.0 * self.xy_extent) / self.xy_res)) + 1
        self.table = np.full((self.x_cells, self.y_cells, self.theta_bins), np.inf, dtype=float)
        self._built = False

    def _meta_dict(self):
        return {
            'xy_res': float(self.xy_res),
            'xy_extent': float(self.xy_extent),
            'theta_bins': int(self.theta_bins),
            'turning_radius': float(self.turning_radius),
            'x_cells': int(self.x_cells),
            'y_cells': int(self.y_cells),
        }

    def _normalize_angle(self, angle):
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def _theta_to_idx(self, theta):
        t = (self._normalize_angle(theta) + math.pi) / (2.0 * math.pi)
        return int(round(t * (self.theta_bins - 1))) % self.theta_bins

    def _xy_to_idx(self, x, y):
        ix = int(round((x + self.xy_extent) / self.xy_res))
        iy = int(round((y + self.xy_extent) / self.xy_res))
        return ix, iy

    def build(self):
        if self._built:
            return

        print('>> 建立 non-holonomic heuristic lookup table ...')
        t0 = time.perf_counter()
        for ix in range(self.x_cells):
            dx = -self.xy_extent + ix * self.xy_res
            for iy in range(self.y_cells):
                dy = -self.xy_extent + iy * self.xy_res
                for it in range(self.theta_bins):
                    dtheta = -math.pi + (2.0 * math.pi * it) / self.theta_bins
                    self.table[ix, iy, it] = rs.path_length((0.0, 0.0, 0.0), (dx, dy, dtheta), self.turning_radius)

        self._built = True
        print(f'>> 查表建立完成，耗時 {time.perf_counter() - t0:.3f} 秒')

    def save_to_npz(self, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        meta = self._meta_dict()
        np.savez_compressed(
            file_path,
            table=self.table,
            xy_res=np.array([meta['xy_res']], dtype=float),
            xy_extent=np.array([meta['xy_extent']], dtype=float),
            theta_bins=np.array([meta['theta_bins']], dtype=np.int32),
            turning_radius=np.array([meta['turning_radius']], dtype=float),
            x_cells=np.array([meta['x_cells']], dtype=np.int32),
            y_cells=np.array([meta['y_cells']], dtype=np.int32),
        )

    def try_load_from_npz(self, file_path):
        if not os.path.exists(file_path):
            return False

        try:
            with np.load(file_path) as data:
                table = data['table']
                meta_loaded = {
                    'xy_res': float(data['xy_res'][0]),
                    'xy_extent': float(data['xy_extent'][0]),
                    'theta_bins': int(data['theta_bins'][0]),
                    'turning_radius': float(data['turning_radius'][0]),
                    'x_cells': int(data['x_cells'][0]),
                    'y_cells': int(data['y_cells'][0]),
                }

            meta_expected = self._meta_dict()
            if meta_loaded != meta_expected:
                return False
            if table.shape != (self.x_cells, self.y_cells, self.theta_bins):
                return False

            self.table = table.astype(float, copy=False)
            self._built = True
            return True
        except Exception:
            return False

    def lookup(self, state, goal_state):
        # Transform state to the goal-centric local frame for translation/rotation invariant lookup.
        dx_w = state.x - goal_state.x
        dy_w = state.y - goal_state.y
        c = math.cos(-goal_state.theta)
        s = math.sin(-goal_state.theta)
        dx = dx_w * c - dy_w * s
        dy = dx_w * s + dy_w * c
        dtheta = self._normalize_angle(state.theta - goal_state.theta)

        ix, iy = self._xy_to_idx(dx, dy)
        it = self._theta_to_idx(dtheta)

        if 0 <= ix < self.x_cells and 0 <= iy < self.y_cells:
            val = float(self.table[ix, iy, it])
            if not np.isfinite(val):
                val = rs.path_length((0.0, 0.0, 0.0), (dx, dy, dtheta), self.turning_radius)
                self.table[ix, iy, it] = val
            return val

        return rs.path_length((0.0, 0.0, 0.0), (dx, dy, dtheta), self.turning_radius)


_NH_TABLE_CACHE = None


def get_nonholonomic_heuristic_table():
    global _NH_TABLE_CACHE
    if _NH_TABLE_CACHE is None:
        _NH_TABLE_CACHE = NonHolonomicHeuristicTable()
        use_disk_cache = bool(int(PARAMS.get('h_table_use_disk_cache', 1)))
        cache_filename = str(PARAMS.get('h_table_cache_filename', 'nh_heuristic_table.npz'))
        cache_path = os.path.join(os.path.dirname(__file__), 'result', cache_filename)

        loaded = False
        if use_disk_cache:
            loaded = _NH_TABLE_CACHE.try_load_from_npz(cache_path)
            if loaded:
                print(f'>> 已載入 non-holonomic 查表快取: {cache_path}')

        if not loaded:
            _NH_TABLE_CACHE.build()
            if use_disk_cache:
                _NH_TABLE_CACHE.save_to_npz(cache_path)
                print(f'>> 已儲存 non-holonomic 查表快取: {cache_path}')
    return _NH_TABLE_CACHE


def get_voronoi_potential(state, obs_map):# 基於 Voronoi 圖的勢能函數，距離障礙物越近勢能越大
    max_v = 0.0
    for cx, cy in state.get_corners():
        d_o = obs_map.get_do(cx, cy)
        if d_o < PARAMS['d_o_max']:
            v = (PARAMS['alpha'] / (PARAMS['alpha'] + d_o)) * ((d_o - PARAMS['d_o_max']) / PARAMS['d_o_max']) ** 2
            max_v = max(max_v, v)
    return max_v


def evaluate_dual_heuristic(state, goal_state, obs_map, nh_table=None):
    # H1: non-holonomic-without-obstacles lookup (懂車不懂路)
    gx, gy = int(state.x / obs_map.grid_res), int(state.y / obs_map.grid_res)

    if obs_map.dijkstra_grid is not None:
        max_x_idx = obs_map.dijkstra_grid.shape[0] - 1
        max_y_idx = obs_map.dijkstra_grid.shape[1] - 1
        gx = max(0, min(gx, max_x_idx))
        gy = max(0, min(gy, max_y_idx))
        dijkstra_cost = obs_map.dijkstra_grid[gx, gy]
    else:
        dijkstra_cost = math.hypot(goal_state.x - state.x, goal_state.y - state.y)

    # H2: obstacle-aware 2D Dijkstra/DP (懂路不懂車)
    if nh_table is not None:
        nh_cost = nh_table.lookup(state, goal_state)
    else:
        qs = (state.x, state.y, state.theta)
        qg = (goal_state.x, goal_state.y, goal_state.theta)
        nh_cost = rs.path_length(qs, qg, PARAMS['turning_radius'])

    return max(dijkstra_cost, nh_cost)


def dynamic_waypoint_optimization(admissible_set, current_state, obs_map, goal_state, lane_heading=-math.pi / 2):
    # Dynamic Optimization Metrics:
    # C1 safety (Voronoi potential), C2 lane parallelism,
    # C3 RS docking safety integral, C4 connectability (heading gap).
    best_wp = None
    min_g = float('inf')

    for s in admissible_set:
        c1 = get_voronoi_potential(s, obs_map)

        # Parallel to lane direction => heading difference close to 0 or pi.
        lane_diff = abs((s.theta - lane_heading + math.pi) % (2.0 * math.pi) - math.pi)
        c2 = min(lane_diff, abs(math.pi - lane_diff)) ** 2

        qs = (s.x, s.y, s.theta)
        qg = (goal_state.x, goal_state.y, goal_state.theta)
        rs_pts = rs.path_sample(qs, qg, PARAMS['turning_radius'], step_size=0.5)
        c3 = sum([get_voronoi_potential(State(p[0], p[1], p[2]), obs_map) for p in rs_pts])

        angle_diff = abs((s.theta - current_state.theta + math.pi) % (2 * math.pi) - math.pi)
        c4 = angle_diff ** 2

        total_g = PARAMS['u1'] * c1 + PARAMS['u2'] * c2 + PARAMS['u3'] * c3 + PARAMS['u4'] * c4
        if total_g < min_g:
            min_g = total_g
            best_wp = s

    return best_wp


def build_admissible_set_around_goal(
    goal_state,
    obs_map,
    x_offsets=(-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0),
    y_offsets=(-5.0, -3.0, -1.5, 0.0, 1.5, 3.0, 5.0),
    heading_offsets=(-math.pi / 2, 0.0, math.pi / 2, math.pi),):
    # Precompute admissible states near the goal that can directly RS-connect without collision.
    c = math.cos(goal_state.theta)
    s = math.sin(goal_state.theta)
    admissible = []
    seen = set()

    for dx in x_offsets:
        for dy in y_offsets:
            wx = goal_state.x + dx * c - dy * s
            wy = goal_state.y + dx * s + dy * c
            for dtheta in heading_offsets:
                theta = (goal_state.theta + dtheta) % (2.0 * math.pi)
                candidate = State(wx, wy, theta)

                key = (round(candidate.x, 2), round(candidate.y, 2), round(candidate.theta, 2))
                if key in seen:
                    continue
                seen.add(key)

                if not is_collision_free(candidate, obs_map):
                    continue
                if not rs_collision_free(candidate, goal_state, obs_map):
                    continue

                admissible.append(candidate)

    admissible.sort(key=lambda st: math.hypot(st.x - goal_state.x, st.y - goal_state.y))
    return admissible


def get_three_closest_wall_points(state, obs_map, samples_per_edge=15):
    corners = state.get_corners()
    part_edges = [
        ('Front', corners[0], corners[1]),
        ('Right', corners[1], corners[2]),
        ('Rear', corners[2], corners[3]),
        ('Left', corners[3], corners[0]),
    ]

    per_part_best = []
    for part_name, p1, p2 in part_edges:
        best = None
        for t in np.linspace(0.0, 1.0, samples_per_edge, endpoint=True):
            sx = p1[0] + t * (p2[0] - p1[0])
            sy = p1[1] + t * (p2[1] - p1[1])
            dists = np.hypot(obs_map.obs_arr[:, 0] - sx, obs_map.obs_arr[:, 1] - sy)
            idx = int(np.argmin(dists))
            wx, wy = obs_map.obs_arr[idx]
            candidate = (float(dists[idx]), (float(sx), float(sy)), (float(wx), float(wy)), part_name)
            if best is None or candidate[0] < best[0]:
                best = candidate
        per_part_best.append(best)

    per_part_best.sort(key=lambda item: item[0])
    return per_part_best[:3]
