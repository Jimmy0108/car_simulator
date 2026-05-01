import copy
import math
import numpy as np

from collision import is_collision_free
from config import PARAMS
from models import State

try:
    from scipy.optimize import minimize
except ImportError:
    minimize = None


def _compute_thetas_from_points(points, gears):
    thetas = np.zeros(len(points), dtype=float)
    if len(points) == 0:
        return thetas
    if len(points) == 1:
        thetas[0] = 0.0
        return thetas

    for i in range(len(points)):
        if i == 0:
            dx = points[1, 0] - points[0, 0]
            dy = points[1, 1] - points[0, 1]
        elif i == len(points) - 1:
            dx = points[-1, 0] - points[-2, 0]
            dy = points[-1, 1] - points[-2, 1]
        else:
            dx = points[i + 1, 0] - points[i - 1, 0]
            dy = points[i + 1, 1] - points[i - 1, 1]

        if gears[i] == -1:
            dx, dy = -dx, -dy
        thetas[i] = math.atan2(float(dy), float(dx))
    return thetas


def _supersample_with_anchors(path_nodes, spacing):
    pts = []
    gears = []
    anchors = []

    pts.append([path_nodes[0].x, path_nodes[0].y])
    gears.append(path_nodes[0].gear)
    anchors.append(True)

    for i in range(len(path_nodes) - 1):
        a = path_nodes[i]
        b = path_nodes[i + 1]
        dx = b.x - a.x
        dy = b.y - a.y
        seg_len = math.hypot(dx, dy)
        steps = max(1, int(math.ceil(seg_len / max(1e-3, spacing))))

        for k in range(1, steps + 1):
            t = k / steps
            x = a.x + t * dx
            y = a.y + t * dy
            pts.append([x, y])
            gears.append(a.gear)
            anchors.append(k == steps)

    return np.array(pts, dtype=float), np.array(gears, dtype=int), np.array(anchors, dtype=bool)


def _optimize_points_cg(points_ref, gears, obs_map, fixed_mask, maxiter, gtol, min_improve_ratio):
    if minimize is None:
        return np.array(points_ref, dtype=float), False, 0.0

    points_ref = np.array(points_ref, dtype=float)
    pts0 = points_ref.copy()

    free_idx = np.where(~fixed_mask)[0]
    if len(free_idx) == 0:
        return pts0, True, 0.0

    x0 = pts0[free_idx].reshape(-1)

    # Stanford CG 平滑器四個核心權重（根據論文 Section 4.3）
    # 目標函數 = w_rho·E_rho + w_o·E_o + w_kappa·E_kappa + w_s·E_s
    w_rho = PARAMS.get('w_rho', 10.0)      # Voronoi 場懲罰權重
    w_o = PARAMS.get('w_o', 50.0)          # 碰撞懲罰權重
    w_kappa = PARAMS.get('w_kappa', 0.5)   # 曲率懲罰權重
    w_s = PARAMS.get('w_s', 0.5)           # 平滑度懲罰權重
    d_max = PARAMS.get('d_max', 1.3)       # 障礙物距離閾值
    kappa_max = PARAMS.get('kappa_max', 0.22)  # 最大允許曲率

    L = PARAMS['vehicle_L']
    W = PARAMS['vehicle_W']
    corner_safe = PARAMS.get('corner_clearance', 0.25)
    offsets = [(L / 2.0, W / 2.0), (L / 2.0, -W / 2.0), (-L / 2.0, -W / 2.0), (-L / 2.0, W / 2.0)]

    def unpack(vec):
        pts = pts0.copy()
        pts[free_idx] = vec.reshape(-1, 2)
        return pts

    def objective(vec):
        pts = unpack(vec)
        cost = 0.0

        # [E1] Data Term: 保留參考路徑的形狀 (防止平滑過度變形)
        data_term = pts[free_idx] - points_ref[free_idx]
        E_data = float(np.sum(data_term * data_term))
        cost += w_rho * E_data

        # [E2] Smoothness Term: 最小化相鄰節點間位移的二階差分 (平滑度)
        # 計算曲率代理：二階有限差分
        if len(pts) > 2:
            sec = pts[:-2] - 2.0 * pts[1:-1] + pts[2:]
            E_smooth = float(np.sum(sec * sec))
            cost += w_s * E_smooth

        # [E3] Collision Penalty: 嚴格懲罰與障礙物的碰撞
        # 搭配距離閾值 d_max 與二次懲罰函數 σ_o
        thetas = _compute_thetas_from_points(pts, gears)
        E_collision = 0.0
        
        for i in range(1, len(pts) - 1):
            px = float(pts[i, 0])
            py = float(pts[i, 1])
            
            # 車輛中心點碰撞檢查
            d_center = obs_map.get_do(px, py)
            if d_center < d_max:
                penalty = (d_max - d_center) ** 2
                E_collision += penalty
            
            # 車身四角碰撞檢查
            cos_t = math.cos(thetas[i])
            sin_t = math.sin(thetas[i])
            for dx, dy in offsets:
                cx = px + dx * cos_t - dy * sin_t
                cy = py + dx * sin_t + dy * cos_t
                d_corner = obs_map.get_do(cx, cy)
                if d_corner < corner_safe:
                    penalty = (corner_safe - d_corner) ** 2
                    E_collision += penalty
        
        cost += w_o * E_collision

        # [E4] Curvature Penalty: 限制瞬間曲率符合車輛轉向極限
        # 搭配最大允許曲率 kappa_max 與二次懲罰函數 σ_kappa
        if len(pts) > 2:
            E_curvature = 0.0
            for i in range(1, len(pts) - 1):
                # 估計曲率：使用三個連續點的圓形擬合
                p_prev = pts[i - 1]
                p_curr = pts[i]
                p_next = pts[i + 1]
                
                dx1 = p_curr[0] - p_prev[0]
                dy1 = p_curr[1] - p_prev[1]
                d1 = math.hypot(dx1, dy1)
                
                dx2 = p_next[0] - p_curr[0]
                dy2 = p_next[1] - p_curr[1]
                d2 = math.hypot(dx2, dy2)
                
                if d1 > 1e-6 and d2 > 1e-6:
                    # 計算角度變化量 (曲率近似)
                    theta1 = math.atan2(dy1, dx1)
                    theta2 = math.atan2(dy2, dx2)
                    dtheta = abs(theta2 - theta1)
                    # 歸一化到 [0, pi]
                    if dtheta > math.pi:
                        dtheta = 2.0 * math.pi - dtheta
                    
                    # 曲率 = dtheta / avg_distance
                    avg_dist = (d1 + d2) / 2.0
                    kappa = dtheta / max(1e-6, avg_dist)
                    
                    if kappa > kappa_max:
                        penalty = (kappa - kappa_max) ** 2
                        E_curvature += penalty
            
            cost += w_kappa * E_curvature

        return cost

    init_cost = objective(x0)
    result = minimize(
        objective,
        x0,
        method='CG',
        options={'maxiter': int(maxiter), 'gtol': float(gtol), 'disp': False},
    )

    candidate_vec = result.x if result.x is not None else x0
    candidate_cost = objective(candidate_vec)
    improvement_ratio = 0.0
    if init_cost > 0.0:
        improvement_ratio = max(0.0, (init_cost - candidate_cost) / init_cost)

    accepted = bool(result.success or improvement_ratio >= float(min_improve_ratio))
    points_opt = unpack(candidate_vec) if accepted else pts0
    return points_opt, accepted, improvement_ratio


def smooth_trajectory(path_nodes, obs_map):
    print('>> 啟動軌跡平滑化後處理...')
    new_path = copy.deepcopy(path_nodes)
    weight_data = 0.4
    weight_smooth = 0.3
    tolerance = 0.05
    change = tolerance

    while change >= tolerance:
        change = 0.0
        for i in range(1, len(path_nodes) - 1):
            for attr in ['x', 'y']:
                old_val = getattr(new_path[i], attr)
                val_data = getattr(path_nodes[i], attr)
                val_prev = getattr(new_path[i - 1], attr)
                val_next = getattr(new_path[i + 1], attr)

                new_val = old_val + weight_data * (val_data - old_val) + weight_smooth * (val_prev + val_next - 2.0 * old_val)
                setattr(new_path[i], attr, new_val)
                change += abs(old_val - new_val)

    return new_path


def smooth_trajectory_cg(path_nodes, obs_map):
    if len(path_nodes) <= 2:
        return copy.deepcopy(path_nodes)

    if minimize is None:
        print('>> 未安裝 scipy，改用原本平滑器。')
        return smooth_trajectory(path_nodes, obs_map)

    print('>> 啟動 CG 軌跡平滑化後處理...')
    new_path = copy.deepcopy(path_nodes)
    ref_xy = np.array([[n.x, n.y] for n in path_nodes], dtype=float)
    x0 = ref_xy[1:-1].reshape(-1)

    # Stanford CG 平滑器四個核心權重（根據論文 Section 4.3）
    # Stage-1：使用標準權重平滑從 A* 得到的粗糙路徑
    w_rho = PARAMS.get('w_rho', 10.0)      # Voronoi 場懲罰權重
    w_o = PARAMS.get('w_o', 50.0)          # 碰撞懲罰權重
    w_kappa = PARAMS.get('w_kappa', 0.5)   # 曲率懲罰權重
    w_s = PARAMS.get('w_s', 0.5)           # 平滑度懲罰權重
    d_max = PARAMS.get('d_max', 1.3)       # 障礙物距離閾值

    def unpack(vec):
        pts = ref_xy.copy()
        pts[1:-1] = vec.reshape(-1, 2)
        return pts

    def objective(vec):
        pts = unpack(vec)

        # [E1] Data Term: 保留原始路徑形狀
        data_term = pts[1:-1] - ref_xy[1:-1]
        cost = w_rho * float(np.sum(data_term * data_term))

        # [E2] Smoothness Term: 最小化二階差分 (平滑度)
        sec = pts[:-2] - 2.0 * pts[1:-1] + pts[2:]
        cost += w_s * float(np.sum(sec * sec))

        # [E3] Collision Penalty: 懲罰與障礙物靠近
        for i, p in enumerate(pts[1:-1]):
            # 中心點懲罰
            d_center = obs_map.get_do(float(p[0]), float(p[1]))
            if d_center < d_max:
                cost += w_o * (d_max - d_center) ** 2
            
            # 車身四角防碰撞懲罰
            approx_theta = path_nodes[i + 1].theta
            L = PARAMS['vehicle_L']
            W = PARAMS['vehicle_W']
            cos_t = math.cos(approx_theta)
            sin_t = math.sin(approx_theta)
            
            offsets = [(L/2, W/2), (L/2, -W/2), (-L/2, -W/2), (-L/2, W/2)]
            
            for dx, dy in offsets:
                cx = float(p[0]) + dx * cos_t - dy * sin_t
                cy = float(p[1]) + dx * sin_t + dy * cos_t
                d_corner = obs_map.get_do(cx, cy)
                
                corner_safe = PARAMS.get('corner_clearance', 0.25)
                if d_corner < corner_safe:
                    cost += (w_o * 2.0) * (corner_safe - d_corner) ** 2

        # [E4] Curvature Penalty: 限制曲率
        # 在 Stage-1 中使用輕度曲率限制
        if len(pts) > 2:
            for i in range(1, len(pts) - 1):
                p_prev = pts[i - 1]
                p_curr = pts[i]
                p_next = pts[i + 1]
                
                dx1 = p_curr[0] - p_prev[0]
                dy1 = p_curr[1] - p_prev[1]
                d1 = math.hypot(dx1, dy1)
                
                dx2 = p_next[0] - p_curr[0]
                dy2 = p_next[1] - p_curr[1]
                d2 = math.hypot(dx2, dy2)
                
                if d1 > 1e-6 and d2 > 1e-6:
                    theta1 = math.atan2(dy1, dx1)
                    theta2 = math.atan2(dy2, dx2)
                    dtheta = abs(theta2 - theta1)
                    if dtheta > math.pi:
                        dtheta = 2.0 * math.pi - dtheta
                    
                    avg_dist = (d1 + d2) / 2.0
                    kappa = dtheta / max(1e-6, avg_dist)
                    
                    # Stage-1 曲率限制較寬鬆
                    kappa_max_stage1 = PARAMS.get('kappa_max', 0.22) * 1.5
                    if kappa > kappa_max_stage1:
                        penalty = (kappa - kappa_max_stage1) ** 2
                        cost += w_kappa * penalty

        return cost

    initial_cost = objective(x0)
    maxiter = int(PARAMS.get('cg_maxiter', 60))
    gtol = float(PARAMS.get('cg_gtol', 1e-3))

    result = minimize(
        objective,
        x0,
        method='CG',
        options={'maxiter': maxiter, 'gtol': gtol, 'disp': False},
    )

    candidate_vec = result.x if result.x is not None else x0
    candidate_cost = objective(candidate_vec)
    improvement_ratio = 0.0
    if initial_cost > 0.0:
        improvement_ratio = max(0.0, (initial_cost - candidate_cost) / initial_cost)

    min_improvement_ratio = float(PARAMS.get('cg_min_improvement_ratio', 0.01))
    accept_candidate = result.success or improvement_ratio >= min_improvement_ratio

    vec_opt = candidate_vec if accept_candidate else x0
    opt_pts = unpack(vec_opt)

    for i in range(1, len(new_path) - 1):
        new_path[i].x = float(opt_pts[i, 0])
        new_path[i].y = float(opt_pts[i, 1])

    if not result.success:
        if accept_candidate:
            print(f'>> [CG] 未完全收斂，但已改善 {improvement_ratio * 100:.1f}% 成本，保留平滑結果。')
        else:
            print('>> [CG] 最佳化未收斂且改善不足，回退原始路徑。')
            return copy.deepcopy(path_nodes)

    for node in new_path:
        if not is_collision_free(node, obs_map):
            print('>> [CG] 平滑後偵測碰撞，回退原始路徑。')
            return copy.deepcopy(path_nodes)

    for i in range(1, len(new_path) - 1):
        dx = new_path[i + 1].x - new_path[i - 1].x
        dy = new_path[i + 1].y - new_path[i - 1].y

        if new_path[i].gear == -1:
            dx, dy = -dx, -dy

        new_path[i].theta = math.atan2(dy, dx)

    if not bool(int(PARAMS.get('cg_enable_stage2', 1))):
        return new_path

    # Stage-2 (Stanford-style): super-sample the polyline and run a second CG pass
    # while keeping original stage-1 vertices fixed as anchors.
    spacing = float(PARAMS.get('cg_stage2_supersample_spacing', 0.35))
    dense_pts, dense_gears, anchor_mask = _supersample_with_anchors(new_path, spacing)
    dense_pts_opt, accepted2, improve2 = _optimize_points_cg(
        dense_pts,
        dense_gears,
        obs_map,
        fixed_mask=anchor_mask,
        maxiter=int(PARAMS.get('cg_stage2_maxiter', 40)),
        gtol=float(PARAMS.get('cg_stage2_gtol', 2e-3)),
        min_improve_ratio=float(PARAMS.get('cg_stage2_min_improvement_ratio', 0.005)),
    )

    if not accepted2:
        print('>> [CG-Stage2] 改善不足，保留第一階段平滑結果。')
        return new_path

    dense_thetas = _compute_thetas_from_points(dense_pts_opt, dense_gears)
    dense_nodes = []
    for i in range(len(dense_pts_opt)):
        dense_nodes.append(
            State(
                float(dense_pts_opt[i, 0]),
                float(dense_pts_opt[i, 1]),
                float(dense_thetas[i]),
                gear=int(dense_gears[i]),
            )
        )

    for node in dense_nodes:
        if not is_collision_free(node, obs_map):
            print('>> [CG-Stage2] 平滑後偵測碰撞，回退第一階段平滑結果。')
            return new_path

    for i in range(1, len(dense_nodes)):
        dense_nodes[i].parent = dense_nodes[i - 1]

    print(f'>> [CG-Stage2] 完成 supersample 曲率優化，改善 {improve2 * 100:.1f}% 成本。')
    return dense_nodes
