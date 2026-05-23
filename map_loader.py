"""Load a parking-lot map from a YAML configuration file.

Usage:
    from map_loader import load_map_from_yaml
    obs_map = load_map_from_yaml('maps/simple_lot.yaml')

The returned object is a ``YamlParkingMap`` instance that fully implements the
``ParkingMap`` abstract interface, so it can be dropped straight into the
existing planner / collision / visualization pipeline with zero changes.
"""

import math
import yaml
import numpy as np

from base_map import ParkingMap


class YamlParkingMap(ParkingMap):
    """A parking map whose layout is loaded entirely from a YAML file."""

    def __init__(self, config: dict):
        super().__init__()

        meta = config.get('metadata', {})
        self.map_name = meta.get('name', 'Unnamed Map')
        self.map_version = meta.get('version', '1.0')

        layout = config['layout']
        self._width = float(layout['width'])
        self._height = float(layout['height'])
        self.grid_res = float(layout.get('grid_resolution', 1.0))

        # 構建障礙物
        self._build_obstacles(config)

        # 方向規則
        self.directional_rules = self._build_directional_rules(config)

        # 可視化配置
        self._viz_config = config.get('visualization', {})

        # 停車場景配置
        self._scenario_config = config.get('scenarios', {})

        # 起始狀態
        self._start_config = config.get('start_state', {
            'x': self._width / 2.0,
            'y': self._height - 1.0,
            'theta': -math.pi / 2,
        })

    # ------------------------------------------------------------------
    # Obstacle construction
    # ------------------------------------------------------------------

    def _build_obstacles(self, config):
        """根據 YAML 中的 obstacles 定義建構障礙物點陣列。"""
        ox, oy = [], []
        obs_defs = config.get('obstacles', [])
        step = float(config.get('layout', {}).get('obstacle_sample_step', 0.5))

        for obs in obs_defs:
            obs_type = obs.get('type', 'line')

            if obs_type == 'line':
                # 兩點之間的線段
                p1 = obs['start']
                p2 = obs['end']
                x1, y1 = float(p1[0]), float(p1[1])
                x2, y2 = float(p2[0]), float(p2[1])
                length = math.hypot(x2 - x1, y2 - y1)
                n_samples = max(2, int(length / step) + 1)
                for t in np.linspace(0, 1, n_samples):
                    ox.append(x1 + t * (x2 - x1))
                    oy.append(y1 + t * (y2 - y1))

            elif obs_type == 'rect':
                # 矩形邊框 (只畫邊)
                x0, y0 = float(obs['origin'][0]), float(obs['origin'][1])
                w, h = float(obs['width']), float(obs['height'])
                # 底
                for x in np.arange(x0, x0 + w + step / 2, step):
                    ox.append(x); oy.append(y0)
                # 頂
                for x in np.arange(x0, x0 + w + step / 2, step):
                    ox.append(x); oy.append(y0 + h)
                # 左
                for y in np.arange(y0, y0 + h + step / 2, step):
                    ox.append(x0); oy.append(y)
                # 右
                for y in np.arange(y0, y0 + h + step / 2, step):
                    ox.append(x0 + w); oy.append(y)

            elif obs_type == 'horizontal_lines':
                # 橫向線段列表，用於車格分隔線
                x_start = float(obs['x_range'][0])
                x_end = float(obs['x_range'][1])
                for y_val in obs['y_values']:
                    y_val = float(y_val)
                    for x in np.arange(x_start, x_end + step / 2, step):
                        ox.append(x)
                        oy.append(y_val)

            elif obs_type == 'vertical_lines':
                # 縱向線段列表，用於車格分隔線
                y_start = float(obs['y_range'][0])
                y_end = float(obs['y_range'][1])
                for x_val in obs['x_values']:
                    x_val = float(x_val)
                    for y in np.arange(y_start, y_end + step / 2, step):
                        ox.append(x_val)
                        oy.append(y)

            elif obs_type == 'points':
                # 離散點列表
                for pt in obs['points']:
                    ox.append(float(pt[0]))
                    oy.append(float(pt[1]))

        self._obstacles = list(zip(ox, oy))
        self._obs_arr = np.array(self._obstacles) if self._obstacles else np.empty((0, 2))

    # ------------------------------------------------------------------
    # Directional rules
    # ------------------------------------------------------------------

    def _build_directional_rules(self, config):
        rules_cfg = config.get('directional_rules', {})
        aisles = []
        for a in rules_cfg.get('aisles', []):
            aisles.append({
                'name': a['name'],
                'x_range': (float(a['x_range'][0]), float(a['x_range'][1])),
                'flow_direction': a.get('flow_direction', 'upward'),
                'forbidden_dy_sign': int(a.get('forbidden_dy_sign', -1)),
            })

        unrestricted = []
        for z in rules_cfg.get('unrestricted_zones', []):
            unrestricted.append({
                'name': z['name'],
                'x_range': (float(z['x_range'][0]), float(z['x_range'][1])),
                'description': z.get('description', ''),
            })

        vp = rules_cfg.get('validation_parameters', {})
        return {
            'aisles': aisles,
            'unrestricted_zones': unrestricted,
            'validation_parameters': {
                'distance_threshold': float(vp.get('distance_threshold', 5.0)),
                'min_reverse_reject': float(vp.get('min_reverse_reject', 0.25)),
                'min_aligned_fraction': float(vp.get('min_aligned_fraction', 0.90)),
                'verbose': bool(vp.get('verbose', False)),
            },
        }

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
            'x_max': self._width,
            'y_min': 0.0,
            'y_max': self._height,
            'width': self._width,
            'height': self._height,
        }

    def get_directional_rules(self):
        if self._directional_rules is not None:
            return self._directional_rules
        return self.directional_rules

    def get_visualization_elements(self):
        """從 YAML visualization 配置轉換為標準化可視化元素。"""
        viz = self._viz_config
        result = {
            'walls': [],
            'slot_lines': [],
            'aisle_guides': [],
            'labels': [],
            'entrance': None,
        }

        for wall in viz.get('walls', []):
            pts = wall['points']
            result['walls'].append({
                'points': [(float(pts[0][0]), float(pts[0][1])),
                           (float(pts[1][0]), float(pts[1][1]))],
                'style': wall.get('style', 'wall'),
            })

        for sl in viz.get('slot_lines', []):
            result['slot_lines'].append({
                'x_range': (float(sl['x_range'][0]), float(sl['x_range'][1])),
                'y_values': [float(y) for y in sl['y_values']],
                'style': sl.get('style', 'env'),
            })

        for vsl in viz.get('v_slot_lines', []):
            if 'v_slot_lines' not in result:
                result['v_slot_lines'] = []
            result['v_slot_lines'].append({
                'y_range': (float(vsl['y_range'][0]), float(vsl['y_range'][1])),
                'x_values': [float(x) for x in vsl['x_values']],
                'style': vsl.get('style', 'env'),
            })

        for ag in viz.get('aisle_guides', []):
            result['aisle_guides'].append({
                'x': float(ag['x']),
                'y_range': (float(ag['y_range'][0]), float(ag['y_range'][1])),
                'style': ag.get('style', 'open'),
            })

        for lb in viz.get('labels', []):
            result['labels'].append({
                'text': lb['text'],
                'x': float(lb['x']),
                'y': float(lb['y']),
                'ha': lb.get('ha', 'center'),
                'va': lb.get('va', 'bottom'),
            })

        ent = viz.get('entrance')
        if ent:
            result['entrance'] = {
                'x_range': (float(ent['x_range'][0]), float(ent['x_range'][1])),
                'y': float(ent['y']),
            }

        return result

    def get_start_state_params(self):
        return {
            'x': float(self._start_config.get('x', self._width / 2.0)),
            'y': float(self._start_config.get('y', self._height - 1.0)),
            'theta': float(self._start_config.get('theta', -math.pi / 2)),
        }

    def get_parking_scenarios_config(self):
        spots = []
        center_wall_x = float(self._scenario_config.get('center_wall_x', self._width / 2.0))

        for s in self._scenario_config.get('spots', []):
            spots.append({
                'name': s['name'],
                'center_x': float(s['center_x']),
                'center_y_options': [float(y) for y in s['center_y_options']],
                'default_y_index': int(s.get('default_y_index', 1)),
                'outward_theta': float(s.get('outward_theta', math.pi if float(s['center_x']) < center_wall_x else 0.0)),
            })

        return {
            'center_wall_x': center_wall_x,
            'spots': spots,
        }


def load_map_from_yaml(yaml_path: str) -> YamlParkingMap:
    """Load and return a YamlParkingMap from the given YAML file path."""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    print(f'>> 已從 YAML 載入地圖: {config.get("metadata", {}).get("name", yaml_path)}')
    return YamlParkingMap(config)
