import math
import heapq
import reeds_shepp as rs

from collision import is_collision_free, rs_collision_free
from config import AISLE_LEFT_CENTER_X, AISLE_RIGHT_CENTER_X, CENTER_WALL_X, PARAMS
from heuristics import (
    build_admissible_set_around_goal,
    dynamic_waypoint_optimization,
    evaluate_dual_heuristic,
    get_nonholonomic_heuristic_table,
    get_voronoi_potential,
)
from map_rules import DirectionalRuleSet
from models import State


RS_STEP_SIZE = 0.25


def expand_nodes_with_rules(state, rule_set, goal_state):
    step_size = PARAMS['step_size']
    turn_angle = math.radians(PARAMS['max_steer_angle'])
    neighbors = []
    for gear in [1, -1]:
        for steer in [0, turn_angle, -turn_angle]:
            ntheta = (state.theta + gear * math.tan(steer) * step_size / PARAMS['wheelbase']) % (2 * math.pi)
            nx = state.x + gear * step_size * math.cos(ntheta)
            ny = state.y + gear * step_size * math.sin(ntheta)

            new_state = State(nx, ny, ntheta, gear=gear, steer=steer, parent=state)
            if not rule_set.is_transition_allowed(state, new_state, goal_state):
                continue

            step_cost = step_size
            if gear == -1:
                step_cost *= PARAMS['reverse_penalty']
            if gear != state.gear:
                step_cost += PARAMS['switch_gear_penalty']

            if steer != 0.0:
                step_cost += PARAMS['steerr_penalty']
            if steer != state.steer:
                step_cost += PARAMS['steer_change_penalty']

            new_state.g_cost = state.g_cost + step_cost
            neighbors.append(new_state)
    return neighbors


def single_stage_hybrid_a_star_directional(
    start_state,
    goal_state,
    obstacle_map,
    rule_set,
    enable_rs_shot=True,
    nh_table=None,
    guidance_set=None,
    lane_heading=-math.pi / 2,
):
    open_heap = []
    close_list = set()
    all_explored = []
    push_id = 0
    dynamic_wp = None

    start_state.f_cost = PARAMS['w1'] * 0 + PARAMS['w2'] * evaluate_dual_heuristic(start_state, goal_state, obstacle_map, nh_table=nh_table)
    heapq.heappush(open_heap, (start_state.f_cost, push_id, start_state))
    step = 0

    while open_heap and step < PARAMS['max_search_steps']:
        step += 1
        _, _, current_node = heapq.heappop(open_heap)

        if not rule_set.is_heading_allowed(current_node):
            continue

        if guidance_set and (dynamic_wp is None or step % int(PARAMS['waypoint_update_interval']) == 0):
            dynamic_wp = dynamic_waypoint_optimization(guidance_set, current_node, obstacle_map, goal_state, lane_heading=lane_heading)

        dist = math.hypot(current_node.x - goal_state.x, current_node.y - goal_state.y)
        angle_diff = abs((current_node.theta - goal_state.theta + math.pi) % (2 * math.pi) - math.pi)

        rs_interval = max(1, int(PARAMS['rs_shot_interval']))
        rs_dist_threshold = float(PARAMS['rs_shot_dist_threshold'])
        if enable_rs_shot and (step % rs_interval == 0 or dist < rs_dist_threshold):
            if rs_collision_free(current_node, goal_state, obstacle_map):
                qs = (current_node.x, current_node.y, current_node.theta)
                qg = (goal_state.x, goal_state.y, goal_state.theta)
                rs_pts = rs.path_sample(qs, qg, PARAMS['turning_radius'], step_size=RS_STEP_SIZE)

                # Validate the entire RS path against directional rules.
                # Use the rule_set's batch validator to detect long reverse/shortcut segments.
                # Apply stricter parameters: min_reverse_reject=0.25m, min_aligned_fraction=0.90
                if rule_set.validate_rs_path(rs_pts, goal_state=goal_state, min_reverse_reject=0.25, min_aligned_fraction=0.90):
                    # Build the State chain to return as the successful path endpoint
                    temp_parent = current_node
                    for pt in rs_pts[1:]:
                        rs_state = State(pt[0], pt[1], pt[2], parent=temp_parent)
                        temp_parent = rs_state

                    print('   [RS Shot + Directional Rules] 提早結束搜尋。')
                    return temp_parent, all_explored

        if dist < 0.85 and angle_diff < math.radians(9):
            if is_collision_free(current_node, obstacle_map):
                return current_node, all_explored

        state_key = f'{round(current_node.x, 0)}_{round(current_node.y, 0)}_{round(current_node.theta, 1)}_{current_node.gear}'
        if state_key in close_list:
            continue
        close_list.add(state_key)
        all_explored.append(current_node)

        for neighbor in expand_nodes_with_rules(current_node, rule_set, goal_state):
            if not is_collision_free(neighbor, obstacle_map):
                continue

            mid_theta = math.atan2(
                (math.sin(current_node.theta) + math.sin(neighbor.theta)) / 2.0,
                (math.cos(current_node.theta) + math.cos(neighbor.theta)) / 2.0,
            )
            mid_state = State((current_node.x + neighbor.x) / 2.0, (current_node.y + neighbor.y) / 2.0, mid_theta)
            if not rule_set.is_heading_allowed(mid_state):
                continue

            if is_collision_free(mid_state, obstacle_map):
                h_cost = evaluate_dual_heuristic(neighbor, goal_state, obstacle_map, nh_table=nh_table)
                v_cost = get_voronoi_potential(neighbor, obstacle_map)

                guidance_bias = 0.0
                if dynamic_wp is not None:
                    guidance_bias = PARAMS['guidance_bias_weight'] * math.hypot(neighbor.x - dynamic_wp.x, neighbor.y - dynamic_wp.y)

                lane_bias = 0.0
                w_lane = float(PARAMS.get('w_lane', 0.5))
                if w_lane > 0.0:
                    d_center = rule_set.get_lane_center_distance(neighbor.x, neighbor.y, obstacle_map)
                    lane_bias = w_lane * (d_center ** 2)

                neighbor.f_cost = (
                    PARAMS['w1'] * neighbor.g_cost
                    + PARAMS['w2'] * h_cost
                    + PARAMS['w3'] * v_cost
                    + guidance_bias
                    + lane_bias
                )
                push_id += 1
                heapq.heappush(open_heap, (neighbor.f_cost, push_id, neighbor))

    return None, all_explored


def multi_stage_planning_directional(start, goal, obs_map):
    # 從停車場地圖讀取方向規則配置
    rules_config = obs_map.get_directional_rules()
    rule_set = DirectionalRuleSet(rules_config=rules_config)  # ← 傳入規則配置
    nh_table = get_nonholonomic_heuristic_table()

    print('>> 建立目標周圍允許集合 (Admissible Set) ...')
    admissible_set = build_admissible_set_around_goal(goal, obs_map)
    if not admissible_set:
        channel_x = AISLE_LEFT_CENTER_X if goal.x < CENTER_WALL_X else AISLE_RIGHT_CENTER_X
        for y_offset in [-3, 0, 3]:
            admissible_set.append(State(channel_x, goal.y + y_offset, -math.pi / 2))
            admissible_set.append(State(channel_x, goal.y + y_offset, math.pi / 2))

    dynamic_wp = dynamic_waypoint_optimization(admissible_set, start, obs_map, goal)
    if dynamic_wp is None:
        dynamic_wp = admissible_set[0]
    print(f'>> [方向規則模式] 鎖定最佳中繼點 X={dynamic_wp.x:.1f}, Y={dynamic_wp.y:.1f}')

    obs_map.precompute_2d_dijkstra(dynamic_wp, directional=True)

    print('>> [階段一] 規劃 入口 -> 動態中繼點 ...')
    node1, exp1 = single_stage_hybrid_a_star_directional(
        start,
        dynamic_wp,
        obs_map,
        rule_set=rule_set,
        enable_rs_shot=True,
        nh_table=nh_table,
        guidance_set=admissible_set,
    )
    if not node1:
        return None, exp1

    obs_map.precompute_2d_dijkstra(goal, directional=True)

    print('>> [階段二] 規劃 中繼點 -> 目標車位 ...')
    start_2 = State(node1.x, node1.y, node1.theta, gear=node1.gear)
    node2, exp2 = single_stage_hybrid_a_star_directional(
        start_2,
        goal,
        obs_map,
        rule_set=rule_set,
        enable_rs_shot=True,
        nh_table=nh_table,
        guidance_set=admissible_set,
    )

    if not node2:
        return None, exp1 + exp2

    curr = node2
    while curr.parent is not None:
        curr = curr.parent
    curr.parent = node1.parent
    return node2, exp1 + exp2
