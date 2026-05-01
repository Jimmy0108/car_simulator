import math
import os
import time

from config import AISLE_RIGHT_CENTER_X, PARAM_STRATEGIES, apply_param_strategy
from models import State
from parking_map import UndergroundParkingMap
from planner import multi_stage_planning
from scenarios import get_quadrant_parking_scenarios, prompt_quadrant_slot_selection
from visualization import plot_results


if __name__ == '__main__':
    run_start_t = time.perf_counter()
    start = State(AISLE_RIGHT_CENTER_X, 44.0, -math.pi / 2)
    obs_map = UndergroundParkingMap()

    output_dir = os.path.join(os.path.dirname(__file__), 'result')
    os.makedirs(output_dir, exist_ok=True)

    selected_slots = prompt_quadrant_slot_selection()
    scenarios = get_quadrant_parking_scenarios(selected_slots)
    total_scenarios = len(scenarios)
    success_count = 0
    successful_planning_times = []

    for idx, (quadrant, orientation, goal) in enumerate(scenarios, start=1):
        scenario_start_t = time.perf_counter()
        print(f'\n===== [{idx}/{total_scenarios}] {quadrant} - {orientation} =====')
        final_node, all_explored = None, []
        used_strategy = 'none'
        for strategy_name, updates in PARAM_STRATEGIES:
            apply_param_strategy(strategy_name, updates)
            final_node, all_explored = multi_stage_planning(start, goal, obs_map)
            if final_node:
                used_strategy = strategy_name
                break

        if not final_node:
            scenario_elapsed = time.perf_counter() - scenario_start_t
            print(f'   [Time] 規劃耗時: {scenario_elapsed:.3f} 秒')
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
        scenario_elapsed = time.perf_counter() - scenario_start_t
        success_count += 1
        successful_planning_times.append(scenario_elapsed)
        print(f'   [Time] 規劃耗時: {scenario_elapsed:.3f} 秒')

    total_elapsed = time.perf_counter() - run_start_t
    if successful_planning_times:
        avg_time = sum(successful_planning_times) / len(successful_planning_times)
        print(f'\n平均生成一段成功路徑耗時: {avg_time:.3f} 秒 ({success_count} 段)')
    else:
        print('\n平均生成一段成功路徑耗時: N/A (0 段成功路徑)')
    print(f'總執行耗時: {total_elapsed:.3f} 秒')

    print(f'\n完成，共輸出 {total_scenarios} 次規劃結果（若個別路徑失敗則不會產圖）。輸出資料夾: {output_dir}')
