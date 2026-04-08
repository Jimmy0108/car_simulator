import math
import os

from config import AISLE_RIGHT_CENTER_X, PARAM_STRATEGIES, apply_param_strategy
from models import State
from parking_map import UndergroundParkingMap
from planner import multi_stage_planning
from scenarios import get_quadrant_parking_scenarios, prompt_quadrant_slot_selection
from visualization import plot_results


if __name__ == '__main__':
    start = State(AISLE_RIGHT_CENTER_X, 44.0, -math.pi / 2)
    obs_map = UndergroundParkingMap()

    output_dir = os.path.join(os.path.dirname(__file__), 'result')
    os.makedirs(output_dir, exist_ok=True)

    selected_slots = prompt_quadrant_slot_selection()
    scenarios = get_quadrant_parking_scenarios(selected_slots)
    for idx, (quadrant, orientation, goal) in enumerate(scenarios, start=1):
        print(f'\n===== [{idx}/8] {quadrant} - {orientation} =====')
        final_node, all_explored = None, []
        used_strategy = 'none'
        for strategy_name, updates in PARAM_STRATEGIES:
            apply_param_strategy(strategy_name, updates)
            final_node, all_explored = multi_stage_planning(start, goal, obs_map)
            if final_node:
                used_strategy = strategy_name
                break

        if not final_node:
            print('   [Failed] 找不到可行路徑')
            continue

        image_name = f'{idx:02d}_{quadrant}_{orientation}.png'
        image_path = os.path.join(output_dir, image_name)
        plot_results(
            start,
            goal,
            final_node,
            all_explored,
            obs_map,
            title_suffix=f'{quadrant} - {orientation} ({used_strategy})',
            save_path=image_path,
            show_plot=False,
        )
        print(f'   [Saved] {image_path}')

    print(f'\n完成，共輸出 8 次規劃結果（若個別路徑失敗則不會產圖）。輸出資料夾: {output_dir}')
