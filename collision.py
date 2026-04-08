import math
import reeds_shepp as rs

from config import PARAMS
from models import State


def is_collision_free(state, obstacle_map):
    # Convert rear-axle state to geometric center for center-distance safety check.
    wb = PARAMS['wheelbase']
    geom_center_x = state.x + (wb / 2.0) * math.cos(state.theta)
    geom_center_y = state.y + (wb / 2.0) * math.sin(state.theta)

    center_safe_dist = (PARAMS['vehicle_W'] / 2.0) + PARAMS['center_clearance_buffer']

    if obstacle_map.get_do(geom_center_x, geom_center_y) < center_safe_dist:
        return False

    for cx, cy in state.get_corners():
        if obstacle_map.get_do(cx, cy) < PARAMS['corner_clearance']:
            return False
    return True

def rs_collision_free(q_start, q_goal, obs_map):
    qs = (q_start.x, q_start.y, q_start.theta)
    qg = (q_goal.x, q_goal.y, q_goal.theta)
    points = rs.path_sample(qs, qg, PARAMS['turning_radius'], step_size=0.5)
    for pt in points:
        tmp_state = State(pt[0], pt[1], pt[2])
        if not is_collision_free(tmp_state, obs_map):
            return False
    return True

def get_collision_marker_points(state, obs_map):
    markers = []

    wb = PARAMS['wheelbase']
    geom_center_x = state.x + (wb / 2.0) * math.cos(state.theta)
    geom_center_y = state.y + (wb / 2.0) * math.sin(state.theta)

    center_safe_dist = (PARAMS['vehicle_W'] / 2.0) + PARAMS['center_clearance_buffer']
    if obs_map.get_do(geom_center_x, geom_center_y) < center_safe_dist:
        markers.append((geom_center_x, geom_center_y))

    for cx, cy in state.get_corners():
        if obs_map.get_do(cx, cy) < PARAMS['corner_clearance']:
            markers.append((cx, cy))

    unique_markers = []
    for mx, my in markers:
        if not any(math.hypot(mx - ux, my - uy) < 0.05 for ux, uy in unique_markers):
            unique_markers.append((mx, my))
    return unique_markers
