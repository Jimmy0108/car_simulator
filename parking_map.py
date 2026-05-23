import math
import numpy as np

from base_map import ParkingMap
from config import (
    CENTER_WALL_X,
    PARKING_SPOT_WIDTH,
    RIGHT_WALL_X,
    ROW1_X_END,
    ROW1_X_START,
    ROW2_X_END,
    ROW2_X_START,
    ROW3_X_END,
    ROW3_X_START,
    ROW4_X_END,
    ROW4_X_START,
    AISLE_RIGHT_CENTER_X,
    LEFT_INNER_SPOT_CENTER_X,
    RIGHT_INNER_SPOT_CENTER_X,
)


class UndergroundParkingMap(ParkingMap):
    def __init__(self):
        super().__init__()
        ox, oy = [], []
        for x in np.arange(0, ROW3_X_END, 0.5):
            ox.append(x)
            oy.append(45.0)
        for x in np.arange(ROW4_X_START, RIGHT_WALL_X + 0.5, 0.5):
            ox.append(x)
            oy.append(45.0)
        for x in np.arange(0, RIGHT_WALL_X + 0.5, 0.5):
            ox.append(x)
            oy.append(0.0)
        for y in np.arange(0, 45.5, 0.5):
            ox.append(0.0)
            oy.append(y)
        for y in np.arange(0, 45.5, 0.5):
            ox.append(RIGHT_WALL_X)
            oy.append(y)
        for y in np.arange(4.5, 36.5, 0.5):
            ox.append(CENTER_WALL_X)
            oy.append(y)

        spot_width, start_y, end_y = PARKING_SPOT_WIDTH, 4.5, 36.5
        y_lines = np.arange(start_y, end_y + spot_width, spot_width)
        row1_x = np.arange(ROW1_X_START, ROW1_X_END + 1e-6, 0.5)
        row2_x = np.arange(ROW2_X_START, ROW2_X_END + 1e-6, 0.5)
        row3_x = np.arange(ROW3_X_START, ROW3_X_END + 1e-6, 0.5)
        row4_x = np.arange(ROW4_X_START, ROW4_X_END + 1e-6, 0.5)
        for y in y_lines:
            for x in row1_x:
                ox.append(x)
                oy.append(y)
            for x in row2_x:
                ox.append(x)
                oy.append(y)
            for x in row3_x:
                ox.append(x)
                oy.append(y)
            for x in row4_x:
                ox.append(x)
                oy.append(y)

        self._obstacles = list(zip(ox, oy))
        self._obs_arr = np.array(self._obstacles)
        
        # 初始化方向規則配置
        self.directional_rules = self._initialize_directional_rules()

    # ------------------------------------------------------------------
    # Abstract interface implementation
    # ------------------------------------------------------------------

    @property
    def obstacles(self):
        return self._obstacles

    @property
    def obs_arr(self):
        return self._obs_arr

    @property
    def map_bounds(self):
        return {
            'x_min': 0.0,
            'x_max': RIGHT_WALL_X,
            'y_min': 0.0,
            'y_max': 45.0,
            'width': RIGHT_WALL_X,
            'height': 45.0,
        }

    def _initialize_directional_rules(self):
        """動態初始化方向規則配置，支援未來的隨機地圖修改。
        
        返回一個字典，包含所有走廊的方向規則和驗證參數。
        """
        return {
            'aisles': [
                {
                    'name': 'left_aisle',
                    'x_range': (ROW1_X_END - 0.5, ROW2_X_START + 0.5),
                    'flow_direction': 'upward',      # 允許向上行駛
                    'forbidden_dy_sign': -1,         # 禁止負Y方向 (向下)
                },
                {
                    'name': 'right_aisle',
                    'x_range': (ROW3_X_END - 0.5, ROW4_X_START + 0.5),
                    'flow_direction': 'downward',    # 允許向下行駛
                    'forbidden_dy_sign': 1,          # 禁止正Y方向 (向上)
                },
            ],
            'unrestricted_zones': [
                {
                    'name': 'center_wall',
                    'x_range': (CENTER_WALL_X - 4.0, CENTER_WALL_X + 4.0),
                    'description': '中央牆壁區域，無方向限制',
                },
            ],
            'validation_parameters': {
                'distance_threshold': 5.0,        # 距離目標超過此值時啟動嚴格檢查
                'min_reverse_reject': 0.25,       # 最大允許逆向位移(米)
                'min_aligned_fraction': 0.90,     # 最少對齊比例(90%)
                'verbose': False,                 # 調試日誌開關
            }
        }

    def get_directional_rules(self):
        """公開接口：獲取方向規則配置。
        
        Returns:
            dict: 方向規則配置，包含走廊定義、限制區域和驗證參數
        """
        if self._directional_rules is not None:
            return self._directional_rules
        return self.directional_rules

    def get_visualization_elements(self):
        start_y, end_y = 4.5, 36.5
        y_lines = list(np.arange(start_y, end_y + PARKING_SPOT_WIDTH, PARKING_SPOT_WIDTH))
        return {
            'walls': [
                {'points': [(0, 45), (ROW3_X_END, 45)], 'style': 'wall'},
                {'points': [(ROW4_X_START, 45), (RIGHT_WALL_X, 45)], 'style': 'wall'},
                {'points': [(0, 0), (RIGHT_WALL_X, 0)], 'style': 'wall'},
                {'points': [(0, 0), (0, 45)], 'style': 'wall'},
                {'points': [(RIGHT_WALL_X, 0), (RIGHT_WALL_X, 45)], 'style': 'wall'},
                {'points': [(CENTER_WALL_X, 4.5), (CENTER_WALL_X, 36.5)], 'style': 'wall'},
            ],
            'slot_lines': [
                {'x_range': (ROW1_X_START, ROW1_X_END), 'y_values': y_lines, 'style': 'env'},
                {'x_range': (ROW2_X_START, ROW2_X_END), 'y_values': y_lines, 'style': 'env'},
                {'x_range': (ROW3_X_START, ROW3_X_END), 'y_values': y_lines, 'style': 'env'},
                {'x_range': (ROW4_X_START, ROW4_X_END), 'y_values': y_lines, 'style': 'env'},
            ],
            'aisle_guides': [
                {'x': ROW1_X_END, 'y_range': (start_y, end_y), 'style': 'open'},
                {'x': ROW2_X_START, 'y_range': (start_y, end_y), 'style': 'open'},
                {'x': ROW3_X_END, 'y_range': (start_y, end_y), 'style': 'open'},
                {'x': ROW4_X_START, 'y_range': (start_y, end_y), 'style': 'open'},
            ],
            'labels': [],
            'entrance': {'x_range': (ROW3_X_END, ROW4_X_START), 'y': 45.0},
        }

    def get_start_state_params(self):
        return {
            'x': AISLE_RIGHT_CENTER_X,
            'y': 44.0,
            'theta': -math.pi / 2,
        }

    def get_parking_scenarios_config(self):
        spot_width = PARKING_SPOT_WIDTH
        start_y, end_y = 4.5, 36.5
        first_center = start_y + spot_width / 2.0
        all_centers = [round(float(y), 2) for y in np.arange(first_center, end_y, spot_width)]
        upper = [y for y in all_centers if y > 22.5]
        lower = [y for y in all_centers if y < 22.5]

        def nearest_idx(values, target):
            return int(np.argmin([abs(v - target) for v in values])) + 1

        return {
            'center_wall_x': CENTER_WALL_X,
            'spots': [
                {
                    'name': 'Q1_upper_left',
                    'center_x': LEFT_INNER_SPOT_CENTER_X,
                    'center_y_options': upper,
                    'default_y_index': nearest_idx(upper, 30.5),
                    'outward_theta': math.pi,
                },
                {
                    'name': 'Q2_upper_right',
                    'center_x': RIGHT_INNER_SPOT_CENTER_X,
                    'center_y_options': upper,
                    'default_y_index': nearest_idx(upper, 30.5),
                    'outward_theta': 0.0,
                },
                {
                    'name': 'Q3_lower_left',
                    'center_x': LEFT_INNER_SPOT_CENTER_X,
                    'center_y_options': lower,
                    'default_y_index': nearest_idx(lower, 14.5),
                    'outward_theta': math.pi,
                },
                {
                    'name': 'Q4_lower_right',
                    'center_x': RIGHT_INNER_SPOT_CENTER_X,
                    'center_y_options': lower,
                    'default_y_index': nearest_idx(lower, 14.5),
                    'outward_theta': 0.0,
                },
            ],
        }
