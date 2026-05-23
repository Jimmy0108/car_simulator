import math

from config import (
    AISLE_LEFT_CENTER_X, 
    AISLE_RIGHT_CENTER_X, 
    CENTER_WALL_X,
    ROW1_X_END,
    ROW2_X_START,
    ROW3_X_END,
    ROW4_X_START
)


class DirectionalRuleSet:
    """Parking-lot lane direction rules.

    支援從外部配置讀取規則，實現規則配置化。
    """

    def __init__(self, forbidden_angle_margin=25.0, rules_config=None):
        """初始化規則集，支援外部規則配置。
        
        Args:
            forbidden_angle_margin (float): 角度邊界(度)
            rules_config (dict): 外部規則配置字典(來自 obs_map.get_directional_rules())
                                 若為 None，使用預設硬編碼規則
        """
        self.margin = math.radians(float(forbidden_angle_margin))
        self.rules_config = rules_config
        
        if rules_config:
            print('>> [DirectionalRuleSet] 使用外部規則配置')
            self.aisles = rules_config.get('aisles', [])
            self.unrestricted_zones = rules_config.get('unrestricted_zones', [])
            self.validation_params = rules_config.get('validation_parameters', {})
        else:
            print('>> [DirectionalRuleSet] 使用預設硬編碼規則')
            self._setup_default_rules()
    
    def _setup_default_rules(self):
        """設定預設硬編碼規則，保持向後兼容。"""
        self.aisles = [
            {
                'name': 'left_aisle',
                'x_range': (ROW1_X_END - 0.5, ROW2_X_START + 0.5),
                'flow_direction': 'upward',
                'forbidden_dy_sign': -1,
            },
            {
                'name': 'right_aisle',
                'x_range': (ROW3_X_END - 0.5, ROW4_X_START + 0.5),
                'flow_direction': 'downward',
                'forbidden_dy_sign': 1,
            },
        ]
        self.unrestricted_zones = [
            {
                'name': 'center_wall',
                'x_range': (CENTER_WALL_X - 4.0, CENTER_WALL_X + 4.0),
            }
        ]
        self.validation_params = {
            'distance_threshold': 5.0,
            'min_reverse_reject': 0.25,
            'min_aligned_fraction': 0.90,
            'verbose': False,
        }

    @staticmethod
    def _normalize_angle(theta):
        # 轉成 -pi 到 pi
        return (theta + math.pi) % (2 * math.pi) - math.pi



    def _in_unrestricted_zone(self, x):
        """檢查是否在無限制區域內。使用動態規則定義。"""
        for zone in self.unrestricted_zones:
            x_min, x_max = zone['x_range']
            if x_min <= x <= x_max:
                return True
        return False
    
    def _get_aisle_info(self, x):
        """獲取位置 x 所在走廊的信息。
        
        Returns:
            dict 或 None: 如果在走廊內返回走廊信息，否則返回 None
        """
        for aisle in self.aisles:
            x_min, x_max = aisle['x_range']
            if x_min <= x <= x_max:
                return aisle
        return None

    def get_lane_center_distance(self, x, y=None, map_ref=None):
        """Return distance to the aisle centerline or skeleton.

        Returns 0 outside aisles or inside unrestricted zones.
        """
        if self._in_unrestricted_zone(x):
            return 0.0

        aisle = self._get_aisle_info(x)
        if not aisle:
            return 0.0

        if map_ref is not None and y is not None and hasattr(map_ref, 'get_skeleton_distance'):
            d_skel = map_ref.get_skeleton_distance(x, y)
            if d_skel is not None:
                return float(d_skel)

        x_min, x_max = aisle['x_range']
        center_x = (x_min + x_max) / 2.0
        return abs(x - center_x)

    def is_heading_allowed(self, state):
        """檢查在該位置的車頭方向是否允許。使用動態規則定義。"""
        if self._in_unrestricted_zone(state.x):
            return True

        aisle = self._get_aisle_info(state.x)
        if not aisle:
            return True  # 不在任何定義的走廊內，允許任意方向
        
        theta = self._normalize_angle(state.theta)
        flow_dir = aisle['flow_direction']
        
        # 根據流向限制車頭方向
        if flow_dir == 'upward':
            # 禁止向下：theta 在 -pi/2 附近
            if (-math.pi + self.margin) < theta < -self.margin:
                return False
        elif flow_dir == 'downward':
            # 禁止向上：theta 在 pi/2 附近
            if self.margin < theta < (math.pi - self.margin):
                return False
        
        return True

    def is_transition_allowed(self, current_state, next_state, goal_state=None, verbose=False):
        """檢查狀態轉換是否允許。使用動態規則定義。"""
        if not self.is_heading_allowed(next_state):
            if verbose or self.validation_params.get('verbose'):
                print(f'      [REJECT] Heading not allowed at ({next_state.x:.2f}, {next_state.y:.2f})')
            return False

        if goal_state is not None:
            dy = next_state.y - current_state.y
            dist_to_goal_y = abs(next_state.y - goal_state.y)
            
            # 獲取動態驗證參數
            distance_threshold = self.validation_params.get('distance_threshold', 5.0)
            verbose_mode = verbose or self.validation_params.get('verbose', False)
            
            # 當距離目標超過閾值時，在走廊內禁止逆向
            if dist_to_goal_y > distance_threshold:
                aisle = self._get_aisle_info(next_state.x)
                if aisle:
                    forbidden_dy_sign = aisle['forbidden_dy_sign']
                    
                    # forbidden_dy_sign = -1 表示禁止向下 (dy < 0)
                    # forbidden_dy_sign = 1 表示禁止向上 (dy > 0)
                    if forbidden_dy_sign == -1 and dy < -1e-3:
                        if verbose_mode:
                            print(f'      [REJECT] {aisle["name"]} reverse: dy={dy:.3f}, dist={dist_to_goal_y:.1f}m')
                        return False
                    elif forbidden_dy_sign == 1 and dy > 1e-3:
                        if verbose_mode:
                            print(f'      [REJECT] {aisle["name"]} reverse: dy={dy:.3f}, dist={dist_to_goal_y:.1f}m')
                        return False

        return True

    def validate_rs_path(self, rs_pts, goal_state=None, min_reverse_reject=None, min_aligned_fraction=None):
        """Validate an entire Reeds-Shepp sampled path (list of (x,y,theta)).
        
        支援動態參數配置。若參數為 None，從 validation_params 讀取。

        - Rejects the path if any sampled segment in a one-way aisle travels
          against the lane flow for more than `min_reverse_reject` meters
          while sufficiently far from the goal.
        - Also ensures each sampled pose's heading is allowed.
        """
        # 從配置讀取參數，若未提供則使用預設值
        if min_reverse_reject is None:
            min_reverse_reject = self.validation_params.get('min_reverse_reject', 0.25)
        if min_aligned_fraction is None:
            min_aligned_fraction = self.validation_params.get('min_aligned_fraction', 0.90)
        
        distance_threshold = self.validation_params.get('distance_threshold', 5.0)
        
        prev_x, prev_y, prev_th = rs_pts[0]
        # cumulative opposite motion per contiguous one-way aisle segment
        cum_opposite = 0.0
        prev_aisle = None

        # statistics for aligned vs total samples in one-way aisles (far from goal)
        aligned_count = 0
        total_count = 0

        for pt in rs_pts[1:]:
            x, y, th = pt
            # heading check
            s = type('S', (), {})()
            s.x, s.y, s.theta = x, y, th
            if not self.is_heading_allowed(s):
                return False

            dy = y - prev_y
            dist_to_goal_y = float('inf')
            if goal_state is not None:
                dist_to_goal_y = abs(y - goal_state.y)

            # determine current aisle
            current_aisle = self._get_aisle_info(x)

            # If aisle changed (entered/exited aisle), reset cumulative counter
            if current_aisle != prev_aisle:
                cum_opposite = 0.0
                prev_aisle = current_aisle

            # if in a one-way aisle and far from goal, accumulate opposite-motion
            if goal_state is not None and dist_to_goal_y > distance_threshold and current_aisle is not None:
                # opposite motion component (positive value when moving against lane flow)
                opposite_motion = 0.0
                aligned = False
                forbidden_dy_sign = current_aisle['forbidden_dy_sign']
                
                if forbidden_dy_sign == -1:
                    # Left aisle: forbidden_dy_sign=-1 means dy < 0 is opposite
                    if dy < -1e-6:
                        opposite_motion = abs(dy)
                        aligned = False
                    else:
                        aligned = True
                elif forbidden_dy_sign == 1:
                    # Right aisle: forbidden_dy_sign=1 means dy > 0 is opposite
                    if dy > 1e-6:
                        opposite_motion = abs(dy)
                        aligned = False
                    else:
                        aligned = True

                cum_opposite += opposite_motion

                # count stats (we count this sample as relevant if any motion happened)
                if abs(dy) > 1e-6:
                    total_count += 1
                    if aligned:
                        aligned_count += 1

                # reject when the contiguous accumulated opposite displacement exceeds threshold
                if cum_opposite > min_reverse_reject:
                    return False

            prev_x, prev_y, prev_th = x, y, th

        # After scanning, ensure that in any far one-way aisle segments the majority
        # of motion samples are aligned with lane flow (protects against many small
        # reverse micro-steps that individually pass threshold but collectively form a shortcut).
        if total_count > 0:
            frac = aligned_count / float(total_count)
            if frac < min_aligned_fraction:
                return False

        return True

    def validate_path_chain(self, node_chain, goal_state=None, verbose=False):
        """Validate a full path represented as a chain of State objects."""
        if not node_chain or len(node_chain) < 2:
            return True, "Path too short to validate"

        violations = []
        prev_state = node_chain[0]

        for current_state in node_chain[1:]:
            if not self.is_heading_allowed(current_state):
                violations.append({
                    'type': 'invalid_heading',
                    'pos': (current_state.x, current_state.y),
                    'theta': current_state.theta,
                    'msg': f'Invalid heading {math.degrees(current_state.theta):.1f}° at ({current_state.x:.2f}, {current_state.y:.2f})'
                })

            if goal_state is not None:
                dy = current_state.y - prev_state.y
                dist_to_goal_y = abs(current_state.y - goal_state.y)

                if dist_to_goal_y > 5.0:
                    aisle = self._get_aisle_info(current_state.x)
                    if aisle:
                        forbidden_dy_sign = aisle['forbidden_dy_sign']
                        if forbidden_dy_sign == -1 and dy < -1e-3:
                            violations.append({
                                'type': 'aisle_reverse',
                                'pos': (current_state.x, current_state.y),
                                'dy': dy,
                                'msg': f"{aisle['name']} reverse motion dy={dy:.3f}m at ({current_state.x:.2f}, {current_state.y:.2f})"
                            })
                        elif forbidden_dy_sign == 1 and dy > 1e-3:
                            violations.append({
                                'type': 'aisle_reverse',
                                'pos': (current_state.x, current_state.y),
                                'dy': dy,
                                'msg': f"{aisle['name']} reverse motion dy={dy:.3f}m at ({current_state.x:.2f}, {current_state.y:.2f})"
                            })

            prev_state = current_state

        if verbose and violations:
            print('  [Path Validation Violations Detected]')
            for v in violations:
                print(f'    - {v["msg"]}')

        return len(violations) == 0, violations
