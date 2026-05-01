import numpy as np


PARAMS = {
    'w1': 1.0,  # 混合A*演算法中的代價函數權重，用於平衡路徑的連續性。
    'w2': 2.0,  # 混合A*演算法中的代價函數權重，用於懲罰轉向。
    'w3': 1.5,  # 混合A*演算法中的代價函數權重，用於懲罰倒車。
    'u1': 100.0,  # Voronoi場的潛在函數權重，用於最大化與障礙物的距離。
    'u2': 10.0,  # 路徑平滑的代價函數權重，與曲率有關。
    'u3': 5.0,   # 路徑平滑的代價函數權重，與曲率變化率有關。
    'u4': 10.0,  # 路徑平滑的代價函數權重，與急動度（jerk）有關。
    'alpha': 5.0,  # Voronoi場的潛在函數中的比例因子，用於調整場的強度。
    'd_o_max': 5.0,  # 在計算Voronoi場時，考慮障礙物的最大距離。
    'vehicle_L': 4.410,  # 車輛長度（米）。
    'vehicle_W': 1.785,  # 車輛寬度（米）。
    'wheelbase': 2.650,  # 車輛軸距（米）。
    'turning_radius': 4.59,  # 車輛的最小轉彎半徑（米）。
    'max_steer_angle': 30.0,  # 車輛的最大轉向角度（度）。
    'reverse_penalty': 3.0,  # 倒車時的懲罰係數。
    'switch_gear_penalty': 10.0,  # 切換行駛方向（前進/後退）時的懲罰。
    'steerr_penalty': 0.1,  # 轉向時的懲罰。
    'steer_change_penalty': 0.5,  # 轉向角度變化時的懲罰。
    'center_clearance_buffer': 0.25,  # 車輛中心點與障礙物之間的安全距離緩衝。
    'corner_clearance': 0.40,  # 車輛四個角落與障礙物之間的安全距離。
    'step_size': 0.5,  # 路徑規劃時每一步的長度（米）。
    'max_search_steps': 50000,  # Hybrid A*搜索演算法的最大搜索步數。
    'rs_shot_interval': 3,  # 週期性 RS 解析展開觸發間隔（每展開 N 個節點嘗試一次）。
    'rs_shot_dist_threshold': 3.0,  # 與目標距離小於此值時，強制嘗試 RS 解析展開。
    'guidance_bias_weight': 0.15,  # 動態中繼點引導偏置權重，越大越偏向中繼點。
    'waypoint_update_interval': 5,  # 在 A* 中重新挑選最佳中繼點的節點展開間隔。
    'h_table_xy_res': 1.0,  # non-holonomic 啟發式查表的 x/y 離散解析度（米）。
    'h_table_xy_extent': 15.0,  # non-holonomic 啟發式查表的 x/y 範圍（±米）。
    'h_table_theta_bins': 36,  # non-holonomic 啟發式查表的朝向離散數。
    'h_table_use_disk_cache': 1,  # 是否啟用 non-holonomic 查表磁碟快取（1=啟用, 0=停用）。
    'h_table_cache_filename': 'nh_heuristic_table.npz',  # 查表快取檔名。
    'cg_maxiter': 1,  # CG 平滑器最大迭代次數。
    'cg_gtol': 1e-4,  # CG 平滑器梯度收斂門檻。
    'cg_min_improvement_ratio': 0.01,  # 若未收斂，至少需要比初始成本改善多少比例才保留結果。
    'cg_enable_stage2': 0,  # 是否啟用第二階段 supersample + CG 曲率優化。
    'cg_stage2_supersample_spacing': 0.35,  # 第二階段超取樣點距（米）。
    'cg_stage2_maxiter': 40,  # 第二階段 CG 最大迭代數。
    'cg_stage2_gtol': 2e-3,  # 第二階段 CG 梯度門檻。
    'cg_stage2_min_improvement_ratio': 0.005,  # 第二階段未收斂時的最小改善比例。
    
    # Stanford CG 平滑器四個核心權重 (根據論文 Section 4.3)
    'w_rho': 10.0,      # $w_\rho$ Voronoi場懲罰權重：將車推離障礙物
    'w_o': 50.0,        # $w_o$ 碰撞懲罰權重：嚴格懲罰與障礙物碰撞，搭配 $d_{max}$ 與二次函數
    'w_kappa': 0.5,     # $w_\kappa$ 曲率懲罰權重：限制瞬間曲率符合非完整約束
    'w_s': 0.5,         # $w_s$ 平滑度懲罰權重：最小化路徑劇烈變化，均勻分佈節點
    'd_max': 1.3,       # $d_{max}$ 障礙物距離閾值（米）：低於此距離時觸發 $w_o$ 懲罰
    'kappa_max': 0.22,  # $\kappa_{max}$ 最大允許曲率（1/轉彎半徑，基於 turning_radius=4.59m）
}

PARAM_STRATEGIES = [
    (
        'paper_strict', # 這是根據論文中描述的參數設定，適合追求高品質路徑的情況，但可能在某些複雜場景下搜尋效率較低。
        {
            'reverse_penalty': 3.0,
            'switch_gear_penalty': 10.0,
            'steerr_penalty': 0.1,
            'steer_change_penalty': 0.5,
            'corner_clearance': 0.25,
            'center_clearance_buffer': 0.10,
            'step_size': 1.0,
            'max_search_steps': 50000,
        },
    ),
    (
        'balanced',# 這是一套經過調整的參數，旨在在保持合理路徑品質的同時提高搜尋效率，適合大多數一般情況。
        {
            'reverse_penalty': 2.6,
            'switch_gear_penalty': 7.0,
            'steerr_penalty': 0.08,
            'steer_change_penalty': 0.35,
            'corner_clearance': 0.30,
            'center_clearance_buffer': 0.15,
            'step_size': 0.9,
            'max_search_steps': 60000,
        },
    ),
    (
        'recovery',# 這套參數更為寬鬆，適合在搜尋失敗或需要快速找到可行解的情況下使用，但生成的路徑可能較為激進或不太平滑。
        {
            'reverse_penalty': 2.2,
            'switch_gear_penalty': 5.0,
            'steerr_penalty': 0.05,
            'steer_change_penalty': 0.20,
            'corner_clearance': 0.25,
            'center_clearance_buffer': 0.10,
            'step_size': 0.8,
            'max_search_steps': 80000,
        },
    ),
    (
        'ultra_relax',# 這套參數是最寬鬆的，適合在需要快速找到可行解或搜尋失敗的情況下使用，但生成的路徑可能較為激進或不太平滑。
        {
            'reverse_penalty': 1.8,
            'switch_gear_penalty': 3.0,
            'steerr_penalty': 0.03,
            'steer_change_penalty': 0.10,
            'corner_clearance': 0.20,
            'center_clearance_buffer': 0.08,
            'step_size': 0.6,
            'max_search_steps': 100000,
        },
    ),
]


def apply_param_strategy(strategy_name, updates):
    for key, value in updates.items():
        PARAMS[key] = value
    print(f'>> 套用參數策略: {strategy_name}')


PARKING_SPOT_LENGTH = 5.50 #車格長度（包含前後安全距離），實際車格長度約4.5m，前後各預留0.5m的安全距離
PARKING_SPOT_WIDTH = 2.50 #車格寬度
AISLE_WIDTH = 8.00 #車道寬度

LEFT_WALL_X = 0.0
ROW1_X_START = LEFT_WALL_X
ROW1_X_END = ROW1_X_START + PARKING_SPOT_LENGTH
ROW2_X_START = ROW1_X_END + AISLE_WIDTH
ROW2_X_END = ROW2_X_START + PARKING_SPOT_LENGTH

CENTER_WALL_X = ROW2_X_END

ROW3_X_START = CENTER_WALL_X
ROW3_X_END = ROW3_X_START + PARKING_SPOT_LENGTH
ROW4_X_START = ROW3_X_END + AISLE_WIDTH
ROW4_X_END = ROW4_X_START + PARKING_SPOT_LENGTH

RIGHT_WALL_X = ROW4_X_END

AISLE_LEFT_CENTER_X = (ROW1_X_END + ROW2_X_START) / 2.0
AISLE_RIGHT_CENTER_X = (ROW3_X_END + ROW4_X_START) / 2.0

LEFT_INNER_SPOT_CENTER_X = (ROW2_X_START + ROW2_X_END) / 2.0
RIGHT_INNER_SPOT_CENTER_X = (ROW3_X_START + ROW3_X_END) / 2.0


def _build_slot_centers(start_y, end_y, spot_width):
    first_center = start_y + spot_width / 2.0
    centers = np.arange(first_center, end_y, spot_width)
    return [round(float(y), 2) for y in centers]


def _nearest_index_1based(values, target):
    idx = int(np.argmin([abs(v - target) for v in values]))
    return idx + 1


_ALL_SLOT_CENTERS_Y = _build_slot_centers(start_y=4.5, end_y=36.5, spot_width=PARKING_SPOT_WIDTH)
_UPPER_SLOT_CENTERS_Y = [y for y in _ALL_SLOT_CENTERS_Y if y > 22.5]
_LOWER_SLOT_CENTERS_Y = [y for y in _ALL_SLOT_CENTERS_Y if y < 22.5]

QUADRANT_SLOT_OPTIONS = {
    'Q1_upper_left': _UPPER_SLOT_CENTERS_Y,
    'Q2_upper_right': _UPPER_SLOT_CENTERS_Y,
    'Q3_lower_left': _LOWER_SLOT_CENTERS_Y,
    'Q4_lower_right': _LOWER_SLOT_CENTERS_Y,
}

DEFAULT_QUADRANT_SLOT_INDEX = {
    'Q1_upper_left': _nearest_index_1based(_UPPER_SLOT_CENTERS_Y, 30.5),
    'Q2_upper_right': _nearest_index_1based(_UPPER_SLOT_CENTERS_Y, 30.5),
    'Q3_lower_left': _nearest_index_1based(_LOWER_SLOT_CENTERS_Y, 14.5),
    'Q4_lower_right': _nearest_index_1based(_LOWER_SLOT_CENTERS_Y, 14.5),
}
