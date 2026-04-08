import copy
import math
import numpy as np

from collision import is_collision_free
from config import PARAMS

try:
    from scipy.optimize import minimize
except ImportError:
    minimize = None


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

    # --- 修正 1：調整平滑器權重與安全距離 ---
    w_data = 10.0      # 增加貼合原始安全路徑的權重 (原本 8.0)
    w_smooth = 0.5     # 降低平滑拉扯的力量 (原本 1.0)
    w_obs = 50.0       # 大幅增加對牆壁的恐懼感 (原本 20.0)
    d_safe = 1.3       # 增加中心點的安全預警距離 (原本 0.8)
    # ----------------------------------------

    def unpack(vec):
        pts = ref_xy.copy()
        pts[1:-1] = vec.reshape(-1, 2)
        return pts

    def objective(vec):
        pts = unpack(vec)

        # 1) 盡量貼近原始路徑
        data_term = pts[1:-1] - ref_xy[1:-1]
        cost = w_data * float(np.sum(data_term * data_term))

        # 2) 二階差分抑制抖動
        sec = pts[:-2] - 2.0 * pts[1:-1] + pts[2:]
        cost += w_smooth * float(np.sum(sec * sec))

        # 3) 靠近障礙物時加懲罰
        for i, p in enumerate(pts[1:-1]):
            # 中心點懲罰
            d_center = obs_map.get_do(float(p[0]), float(p[1]))
            if d_center < d_safe:
                cost += w_obs * (d_safe - d_center) ** 2
            
            # --- 修正 2：補上車身四角防碰撞懲罰 ---
            # 由於平滑器過程中無法即時取得準確的 theta 角，
            # 我們借用原本 path_nodes 的 theta 來估算角落位置
            approx_theta = path_nodes[i + 1].theta
            L = PARAMS['vehicle_L']
            W = PARAMS['vehicle_W']
            cos_t = math.cos(approx_theta)
            sin_t = math.sin(approx_theta)
            
            # 四個角落相對於車輛中心的偏移量（局部坐標）
            offsets = [(L/2, W/2), (L/2, -W/2), (-L/2, -W/2), (-L/2, W/2)]
            
            for dx, dy in offsets:
                # 轉換到全域坐標系
                cx = float(p[0]) + dx * cos_t - dy * sin_t
                cy = float(p[1]) + dx * sin_t + dy * cos_t
                d_corner = obs_map.get_do(cx, cy)
                
                # Use actual corner clearance from config for stricter smoothing
                # Match the checking standard to avoid over-relaxation in tight slots
                corner_safe = PARAMS.get('corner_clearance', 0.25)
                if d_corner < corner_safe:
                    cost += (w_obs * 2.0) * (corner_safe - d_corner) ** 2
            # ------------------------------------

        return cost

    result = minimize(
        objective,
        x0,
        method='CG',
        options={'maxiter': 180, 'gtol': 1e-4, 'disp': False},
    )

    vec_opt = result.x if result.success else x0
    opt_pts = unpack(vec_opt)

    for i in range(1, len(new_path) - 1):
        new_path[i].x = float(opt_pts[i, 0])
        new_path[i].y = float(opt_pts[i, 1])

    if not result.success:
        print('>> [CG] 最佳化未收斂，回退原始路徑。')
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

    return new_path
