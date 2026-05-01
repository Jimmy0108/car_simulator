import math

from config import (
    CENTER_WALL_X,
    DEFAULT_QUADRANT_SLOT_INDEX,
    LEFT_INNER_SPOT_CENTER_X,
    PARAMS,
    QUADRANT_SLOT_OPTIONS,
    RIGHT_INNER_SPOT_CENTER_X,
)
from models import State


def prompt_quadrant_slot_selection():
    print('\n=== 停車格選擇（四象限各選一格）===')
    selection = {}
    ordered_quadrants = [
        'Q1_upper_left',
        'Q2_upper_right',
        'Q3_lower_left',
        'Q4_lower_right',
    ]

    for quadrant in ordered_quadrants:
        options = QUADRANT_SLOT_OPTIONS[quadrant]
        default_idx = DEFAULT_QUADRANT_SLOT_INDEX[quadrant]
        print(f'\n{quadrant} 可選車格中心 y：')
        for idx, y in enumerate(options, start=1):
            marker = ' (default)' if idx == default_idx else ''
            print(f'  {idx}. y={y}{marker}')

        while True:
            try:
                raw = input(f'請輸入 {quadrant} 的格位編號 [1-{len(options)}] (Enter=default): ').strip()
            except EOFError:
                raw = ''

            if raw == '':
                selection[quadrant] = options[default_idx - 1]
                break

            if raw.isdigit():
                chosen_idx = int(raw)
                if 1 <= chosen_idx <= len(options):
                    selection[quadrant] = options[chosen_idx - 1]
                    break

            print('輸入無效，請重新輸入。')

    print('\n=== 已選擇車格 ===')
    for quadrant in ordered_quadrants:
        print(f'{quadrant}: y={selection[quadrant]}')
    return selection


def get_quadrant_parking_scenarios(selected_y_by_quadrant):
    left_center_x = LEFT_INNER_SPOT_CENTER_X
    right_center_x = RIGHT_INNER_SPOT_CENTER_X
    spots = [
        ('Q1_upper_left', left_center_x, selected_y_by_quadrant['Q1_upper_left']),
        ('Q2_upper_right', right_center_x, selected_y_by_quadrant['Q2_upper_right']),
        ('Q3_lower_left', left_center_x, selected_y_by_quadrant['Q3_lower_left']),
        ('Q4_lower_right', right_center_x, selected_y_by_quadrant['Q4_lower_right']),
    ]

    scenarios = []
    wb = PARAMS['wheelbase']
    for quadrant, gx, gy in spots:
        outward_theta = math.pi if gx < CENTER_WALL_X else 0.0

        # Convert desired geometric-center parking spot to rear-axle goal state.
        # Outward-only target: rear-in parking while parked car points toward aisle.
        out_goal_x = gx - (wb / 2.0) * math.cos(outward_theta)
        out_goal_y = gy - (wb / 2.0) * math.sin(outward_theta)

        scenarios.append((quadrant, 'outward', State(out_goal_x, out_goal_y, outward_theta)))
    return scenarios
