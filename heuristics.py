import math
import numpy as np
import reeds_shepp as rs

from config import PARAMS
from models import State


def get_voronoi_potential(state, obs_map):# 基於 Voronoi 圖的勢能函數，距離障礙物越近勢能越大
    max_v = 0.0
    for cx, cy in state.get_corners():
        d_o = obs_map.get_do(cx, cy)
        if d_o < PARAMS['d_o_max']:
            v = (PARAMS['alpha'] / (PARAMS['alpha'] + d_o)) * ((d_o - PARAMS['d_o_max']) / PARAMS['d_o_max']) ** 2
            max_v = max(max_v, v)
    return max_v


def evaluate_dual_heuristic(state, goal_state, obs_map):# 結合 Voronoi potential 和 Reeds-Shepp 路徑長度的啟發式函數
    gx, gy = int(state.x / obs_map.grid_res), int(state.y / obs_map.grid_res)

    if obs_map.dijkstra_grid is not None:
        max_x_idx = obs_map.dijkstra_grid.shape[0] - 1
        max_y_idx = obs_map.dijkstra_grid.shape[1] - 1
        gx = max(0, min(gx, max_x_idx))
        gy = max(0, min(gy, max_y_idx))
        dijkstra_cost = obs_map.dijkstra_grid[gx, gy]
    else:
        dijkstra_cost = math.hypot(goal_state.x - state.x, goal_state.y - state.y)

    qs = (state.x, state.y, state.theta)
    qg = (goal_state.x, goal_state.y, goal_state.theta)
    rs_cost = rs.path_length(qs, qg, PARAMS['turning_radius']) * 1.2

    return max(dijkstra_cost, rs_cost)


def dynamic_waypoint_optimization(admissible_set, current_state, obs_map, goal_state):# 在多階段規劃中，根據當前狀態和障礙物分布動態選擇下一個最佳的中繼點
    best_wp = None
    min_g = float('inf')

    for s in admissible_set:
        g1 = get_voronoi_potential(s, obs_map)

        s_dx = State(s.x + 0.1, s.y, s.theta)
        s_dy = State(s.x, s.y + 0.1, s.theta)
        grad_x = (get_voronoi_potential(s_dx, obs_map) - g1) / 0.1
        grad_y = (get_voronoi_potential(s_dy, obs_map) - g1) / 0.1
        g2 = math.hypot(grad_x, grad_y)

        qs = (s.x, s.y, s.theta)
        qg = (goal_state.x, goal_state.y, goal_state.theta)
        rs_pts = rs.path_sample(qs, qg, PARAMS['turning_radius'], step_size=0.5)
        g3 = sum([get_voronoi_potential(State(p[0], p[1], p[2]), obs_map) for p in rs_pts])

        angle_diff = abs((s.theta - current_state.theta + math.pi) % (2 * math.pi) - math.pi)
        g4 = angle_diff ** 2

        total_g = PARAMS['u1'] * g1 + PARAMS['u2'] * g2 + PARAMS['u3'] * g3 + PARAMS['u4'] * g4
        if total_g < min_g:
            min_g = total_g
            best_wp = s

    return best_wp


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
