import math
import heapq
import reeds_shepp as rs

from collision import is_collision_free, rs_collision_free
from config import (
    AISLE_LEFT_CENTER_X,
    AISLE_RIGHT_CENTER_X,
    CENTER_WALL_X,
    PARAMS,
)
from heuristics import (
    dynamic_waypoint_optimization,
    evaluate_dual_heuristic,
    get_voronoi_potential,
)
from models import State


RS_STEP_SIZE = 0.5


def expand_nodes(state):
    step_size = PARAMS['step_size']
    turn_angle = math.radians(PARAMS['max_steer_angle'])
    neighbors = []
    for gear in [1, -1]:
        for steer in [0, turn_angle, -turn_angle]:
            ntheta = (state.theta + gear * math.tan(steer) * step_size / PARAMS['wheelbase']) % (2 * math.pi)
            nx = state.x + gear * step_size * math.cos(ntheta)
            ny = state.y + gear * step_size * math.sin(ntheta)

            new_state = State(nx, ny, ntheta, gear=gear, steer=steer, parent=state)
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


def single_stage_hybrid_a_star(start_state, goal_state, obstacle_map, enable_rs_shot=True):
    open_heap = []
    close_list = set()
    all_explored = []
    push_id = 0

    start_state.f_cost = PARAMS['w1'] * 0 + PARAMS['w2'] * evaluate_dual_heuristic(start_state, goal_state, obstacle_map)
    heapq.heappush(open_heap, (start_state.f_cost, push_id, start_state))
    step = 0

    while open_heap and step < PARAMS['max_search_steps']:
        step += 1
        _, _, current_node = heapq.heappop(open_heap)

        dist = math.hypot(current_node.x - goal_state.x, current_node.y - goal_state.y)
        angle_diff = abs((current_node.theta - goal_state.theta + math.pi) % (2 * math.pi) - math.pi)

        # More aggressive RS shot: try more frequently and when close to goal
        if enable_rs_shot and (step % 3 == 0 or dist < 3.0):
            if rs_collision_free(current_node, goal_state, obstacle_map):
                print('   [RS Shot] 發現無碰撞 RS 曲線！提早結束搜尋。')
                qs = (current_node.x, current_node.y, current_node.theta)
                qg = (goal_state.x, goal_state.y, goal_state.theta)
                rs_pts = rs.path_sample(qs, qg, PARAMS['turning_radius'], step_size=RS_STEP_SIZE)

                temp_parent = current_node
                for pt in rs_pts[1:]:
                    rs_state = State(pt[0], pt[1], pt[2], parent=temp_parent)
                    temp_parent = rs_state
                return temp_parent, all_explored

        # Goal acceptance: use current_node directly without modifying goal state
        if dist < 0.85 and angle_diff < math.radians(9):
            if is_collision_free(current_node, obstacle_map):
                print(f'   [Goal Accepted] dist={dist:.2f}m, angle_diff={math.degrees(angle_diff):.1f}° ✓')
                return current_node, all_explored

        state_key = f'{round(current_node.x, 0)}_{round(current_node.y, 0)}_{round(current_node.theta, 1)}_{current_node.gear}'
        if state_key in close_list:
            continue
        close_list.add(state_key)
        all_explored.append(current_node)

        for neighbor in expand_nodes(current_node):
            # 1. 檢查目標點是否發生碰撞
            if not is_collision_free(neighbor, obstacle_map):
                continue
                
            # 2. 檢查中間點 (Edge Collision Check)，避免大步長直接跨越障礙物邊角
            mid_theta = math.atan2((math.sin(current_node.theta) + math.sin(neighbor.theta)) / 2.0, 
                                   (math.cos(current_node.theta) + math.cos(neighbor.theta)) / 2.0)
            mid_state = State((current_node.x + neighbor.x) / 2.0, (current_node.y + neighbor.y) / 2.0, mid_theta)
            
            if is_collision_free(mid_state, obstacle_map):
                h_cost = evaluate_dual_heuristic(neighbor, goal_state, obstacle_map)
                v_cost = get_voronoi_potential(neighbor, obstacle_map)
                neighbor.f_cost = PARAMS['w1'] * neighbor.g_cost + PARAMS['w2'] * h_cost + PARAMS['w3'] * v_cost
                push_id += 1
                heapq.heappush(open_heap, (neighbor.f_cost, push_id, neighbor))

    return None, all_explored


def multi_stage_planning(start, goal, obs_map):
    print('>> 建立目標周圍允許集合 (Admissible Set) ...')
    admissible_set = []
    channel_x = AISLE_LEFT_CENTER_X if goal.x < CENTER_WALL_X else AISLE_RIGHT_CENTER_X
    for y_offset in [-3, 0, 3]:
        admissible_set.append(State(channel_x, goal.y + y_offset, -math.pi / 2))
        admissible_set.append(State(channel_x, goal.y + y_offset, math.pi / 2))

    dynamic_wp = dynamic_waypoint_optimization(admissible_set, start, obs_map, goal)
    if dynamic_wp is None:
        print('>> [動態最佳化] 找不到最佳中繼點，改用預設中繼點。')
        dynamic_wp = admissible_set[0]
    print(f'>> [動態最佳化] 鎖定最佳中繼點 X={dynamic_wp.x:.1f}, Y={dynamic_wp.y:.1f}')

    obs_map.precompute_2d_dijkstra(dynamic_wp)

    print('>> [階段一] 規劃 入口 -> 動態中繼點 ...')
    node1, exp1 = single_stage_hybrid_a_star(start, dynamic_wp, obs_map, enable_rs_shot=True)
    if not node1:
        return None, exp1

    obs_map.precompute_2d_dijkstra(goal)

    print('>> [階段二] 規劃 中繼點 -> 目標車位 ...')
    start_2 = State(node1.x, node1.y, node1.theta, gear=node1.gear)
    node2, exp2 = single_stage_hybrid_a_star(start_2, goal, obs_map, enable_rs_shot=True)

    if not node2:
        return None, exp1 + exp2

    curr = node2
    while curr.parent is not None:
        curr = curr.parent
    curr.parent = node1.parent
    return node2, exp1 + exp2
