import math
import os
import glob
import time

from config import PARAM_STRATEGIES, apply_param_strategy, PARAMS
from models import State
from parking_map import UndergroundParkingMap
from planner import multi_stage_planning
from planner_directional import multi_stage_planning_directional
from visualization import plot_results


def prompt_map_selection():
    """讓使用者選擇地圖：內建地圖 或 YAML 自訂地圖。"""
    print('\n=== 地圖選擇 ===')
    print('  1. 地下停車場 (內建 UndergroundParkingMap)')

    # 掃描 maps/ 目錄中的 YAML 檔案
    maps_dir = os.path.join(os.path.dirname(__file__), 'maps')
    yaml_files = []
    if os.path.isdir(maps_dir):
        yaml_files = sorted(glob.glob(os.path.join(maps_dir, '*.yaml')))
        yaml_files += sorted(glob.glob(os.path.join(maps_dir, '*.yml')))

    for i, yf in enumerate(yaml_files, start=2):
        basename = os.path.basename(yf)
        # 嘗試讀取 metadata.name
        try:
            import yaml
            with open(yf, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f)
            map_name = cfg.get('metadata', {}).get('name', basename)
        except Exception:
            map_name = basename
        print(f'  {i}. [YAML] {map_name}  ({basename})')

    max_choice = 1 + len(yaml_files)
    while True:
        try:
            raw = input(f'請選擇地圖 [1-{max_choice}] (Enter=1): ').strip()
        except EOFError:
            raw = ''

        if raw == '' or raw == '1':
            return UndergroundParkingMap()

        if raw.isdigit():
            choice = int(raw)
            if 2 <= choice <= max_choice:
                yaml_path = yaml_files[choice - 2]
                from map_loader import load_map_from_yaml
                return load_map_from_yaml(yaml_path)

        print('輸入無效，請重新輸入。')


def prompt_planner_mode():
    print('\n=== 規劃模式選擇 ===')
    print('  1. 無規則（目前版本）')
    print('  2. 有方向規則（停車場單向車道）')
    while True:
        try:
            raw = input('請輸入模式編號 [1-2] (Enter=1): ').strip()
        except EOFError:
            raw = ''

        if raw == '' or raw == '1':
            return 'classic', multi_stage_planning
        if raw == '2':
            return 'directional', multi_stage_planning_directional
        print('輸入無效，請重新輸入。')


def prompt_scenario_selection(obs_map):
    """根據地圖的 scenarios 配置讓使用者選擇停車格。"""
    config = obs_map.get_parking_scenarios_config()
    spots = config['spots']
    center_wall_x = config['center_wall_x']
    wb = PARAMS['wheelbase']

    print('\n=== 停車格選擇 ===')
    selection = {}

    for spot in spots:
        name = spot['name']
        options = spot['center_y_options']
        default_idx = spot.get('default_y_index', 1)
        print(f'\n{name} 可選車格中心 y：')
        for idx, y in enumerate(options, start=1):
            marker = ' (default)' if idx == default_idx else ''
            print(f'  {idx}. y={y}{marker}')

        while True:
            try:
                raw = input(f'請輸入 {name} 的格位編號 [1-{len(options)}] (Enter=default): ').strip()
            except EOFError:
                raw = ''

            if raw == '':
                selection[name] = options[default_idx - 1]
                break
            if raw.isdigit():
                chosen = int(raw)
                if 1 <= chosen <= len(options):
                    selection[name] = options[chosen - 1]
                    break
            print('輸入無效，請重新輸入。')

    print('\n=== 已選擇車格 ===')
    for spot in spots:
        print(f'{spot["name"]}: y={selection[spot["name"]]}')

    # 組裝 scenarios
    scenarios = []
    for spot in spots:
        name = spot['name']
        gx = spot['center_x']
        gy = selection[name]
        outward_theta = spot['outward_theta']

        out_goal_x = gx - (wb / 2.0) * math.cos(outward_theta)
        out_goal_y = gy - (wb / 2.0) * math.sin(outward_theta)
        scenarios.append((name, 'outward', State(out_goal_x, out_goal_y, outward_theta)))

    return scenarios


if __name__ == '__main__':
    run_start_t = time.perf_counter()

    # 1. 選擇地圖
    obs_map = prompt_map_selection()

    # 2. 從地圖獲取起始狀態
    sp = obs_map.get_start_state_params()
    start = State(sp['x'], sp['y'], sp['theta'])

    # 3. 選擇規劃模式
    mode_name, planning_func = prompt_planner_mode()
    print(f'>> 使用規劃模式: {mode_name}')

    output_dir = os.path.join(os.path.dirname(__file__), 'result')
    os.makedirs(output_dir, exist_ok=True)

    # 4. 選擇停車格 (使用地圖自帶的 scenarios 配置)
    scenarios = prompt_scenario_selection(obs_map)
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
            final_node, all_explored = planning_func(start, goal, obs_map)
            if final_node:
                used_strategy = strategy_name
                break

        if not final_node:
            scenario_elapsed = time.perf_counter() - scenario_start_t
            print(f'   [Time] 規劃耗時: {scenario_elapsed:.3f} 秒')
            print('   [Failed] 找不到可行路徑')
            continue

        image_name = f'{idx:02d}_{quadrant}_{orientation}_{mode_name}.png'
        image_path = os.path.join(output_dir, image_name)
        plot_results(
            start,
            goal,
            final_node,
            all_explored,
            obs_map,
            title_suffix=f'{quadrant} - {orientation} ({mode_name}, {used_strategy})',
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
