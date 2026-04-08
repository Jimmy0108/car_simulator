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
