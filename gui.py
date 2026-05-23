import math
import os
import glob
import time
import traceback

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from config import (
    PARAMS,
    PARAM_STRATEGIES,
    apply_param_strategy,
)
from models import State
from parking_map import UndergroundParkingMap
from map_loader import load_map_from_yaml
from planner import multi_stage_planning
from planner_directional import multi_stage_planning_directional
from visualization import plot_results


class PlannerGui(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Hybrid A* Parameter Console')
        self.resize(1400, 900)

        self.param_inputs = {}
        self.slot_inputs = {}
        self.result_images = []  # Store paths of generated images
        self.current_obs_map = None
        self.current_scenarios_config = None

        root = QWidget()
        self.setCentralWidget(root)
        main_layout = QVBoxLayout(root)

        top_layout = QHBoxLayout()
        main_layout.addLayout(top_layout)

        top_layout.addWidget(self._build_params_panel(), stretch=2)
        top_layout.addWidget(self._build_scenario_panel(), stretch=1)

        btn_layout = QHBoxLayout()
        self.run_btn = QPushButton('Run Planning')
        self.run_btn.clicked.connect(self.run_planning)
        btn_layout.addWidget(self.run_btn)

        self.output_dir_input = QLineEdit(os.path.join(os.path.dirname(__file__), 'result'))
        btn_layout.addWidget(QLabel('Output Folder:'))
        btn_layout.addWidget(self.output_dir_input)

        main_layout.addLayout(btn_layout)

        # Bottom section: Log + Image Preview
        bottom_layout = QHBoxLayout()

        # Log box on the left
        log_group = QGroupBox('Execution Log')
        log_layout = QVBoxLayout(log_group)
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        log_layout.addWidget(self.log_box)
        bottom_layout.addWidget(log_group, stretch=1)

        # Image preview on the right
        image_group = QGroupBox('Result Preview')
        image_layout = QVBoxLayout(image_group)

        # Result selector combo
        selector_layout = QHBoxLayout()
        selector_layout.addWidget(QLabel('Select Result:'))
        self.result_combo = QComboBox()
        self.result_combo.currentIndexChanged.connect(self._update_preview)
        selector_layout.addWidget(self.result_combo)
        image_layout.addLayout(selector_layout)

        # Image label with scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        scroll.setWidget(self.image_label)
        image_layout.addWidget(scroll)

        bottom_layout.addWidget(image_group, stretch=1)
        main_layout.addLayout(bottom_layout, stretch=1)

    def _build_params_panel(self):
        group = QGroupBox('PARAMS (Editable)')
        layout = QVBoxLayout(group)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        form = QFormLayout(content)

        numeric_keys = [
            'w1', 'w2', 'w3', 'u1', 'u2', 'u3', 'u4', 'alpha', 'd_o_max',
            'vehicle_L', 'vehicle_W', 'wheelbase', 'turning_radius',
            'max_steer_angle', 'reverse_penalty', 'switch_gear_penalty',
            'steerr_penalty', 'steer_change_penalty', 'center_clearance_buffer',
            'corner_clearance', 'step_size', 'max_search_steps',
        ]

        for key in numeric_keys:
            if key == 'max_search_steps':
                w = QSpinBox()
                w.setRange(100, 500000)
                w.setValue(int(PARAMS[key]))
            else:
                w = QDoubleSpinBox()
                w.setRange(-100000.0, 100000.0)
                w.setDecimals(4)
                w.setSingleStep(0.1)
                w.setValue(float(PARAMS[key]))
            self.param_inputs[key] = w
            form.addRow(QLabel(key), w)

        scroll.setWidget(content)
        layout.addWidget(scroll)
        return group

    def _build_scenario_panel(self):
        group = QGroupBox('Scenario Setup')
        layout = QVBoxLayout(group)

        map_group = QGroupBox('Map Selection')
        map_layout = QVBoxLayout(map_group)
        self.map_combo = QComboBox()
        self.map_combo.addItem('地下停車場 (內建 UndergroundParkingMap)', 'builtin')

        maps_dir = os.path.join(os.path.dirname(__file__), 'maps')
        if os.path.isdir(maps_dir):
            for yf in sorted(glob.glob(os.path.join(maps_dir, '*.yaml')) + glob.glob(os.path.join(maps_dir, '*.yml'))):
                basename = os.path.basename(yf)
                self.map_combo.addItem(f'[YAML] {basename}', yf)

        self.map_combo.currentIndexChanged.connect(self._on_map_changed)
        map_layout.addWidget(self.map_combo)
        layout.addWidget(map_group)

        start_form = QFormLayout()
        self.start_x_input = QDoubleSpinBox()
        self.start_x_input.setRange(-1000.0, 1000.0)
        self.start_x_input.setDecimals(3)

        self.start_y_input = QDoubleSpinBox()
        self.start_y_input.setRange(-1000.0, 1000.0)
        self.start_y_input.setDecimals(3)

        self.start_theta_input = QDoubleSpinBox()
        self.start_theta_input.setRange(-6.3, 6.3)
        self.start_theta_input.setDecimals(4)

        start_form.addRow('Start X', self.start_x_input)
        start_form.addRow('Start Y', self.start_y_input)
        start_form.addRow('Start Theta(rad)', self.start_theta_input)
        layout.addLayout(start_form)

        self.slot_group = QGroupBox('Target Spots')
        self.slot_layout = QGridLayout(self.slot_group)
        layout.addWidget(self.slot_group)

        strategy_group = QGroupBox('Strategy Order')
        strategy_layout = QVBoxLayout(strategy_group)
        self.strategy_label = QLabel(' -> '.join([name for name, _ in PARAM_STRATEGIES]))
        self.strategy_label.setWordWrap(True)
        strategy_layout.addWidget(self.strategy_label)
        layout.addWidget(strategy_group)

        mode_group = QGroupBox('Planner Mode')
        mode_layout = QVBoxLayout(mode_group)
        self.mode_combo = QComboBox()
        self.mode_combo.addItem('No Rules (Classic)', 'classic')
        self.mode_combo.addItem('Directional Rules (Parking Lot)', 'directional')
        mode_layout.addWidget(self.mode_combo)
        layout.addWidget(mode_group)

        # Trigger initial load
        self._on_map_changed()

        return group

    def _on_map_changed(self):
        try:
            map_data = self.map_combo.currentData()
            if map_data == 'builtin':
                self.current_obs_map = UndergroundParkingMap()
            else:
                self.current_obs_map = load_map_from_yaml(map_data)

            sp = self.current_obs_map.get_start_state_params()
            self.start_x_input.setValue(float(sp['x']))
            self.start_y_input.setValue(float(sp['y']))
            self.start_theta_input.setValue(float(sp['theta']))

            self.current_scenarios_config = self.current_obs_map.get_parking_scenarios_config()

            # Clear old slots
            while self.slot_layout.count():
                child = self.slot_layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
            self.slot_inputs.clear()

            spots = self.current_scenarios_config.get('spots', [])
            for idx, spot in enumerate(spots):
                combo = QComboBox()
                options = spot['center_y_options']
                for y in options:
                    combo.addItem(f'y={y}', y)
                default_idx = spot.get('default_y_index', 1) - 1
                combo.setCurrentIndex(max(0, min(default_idx, combo.count() - 1)))
                self.slot_inputs[spot['name']] = combo
                self.slot_layout.addWidget(QLabel(spot['name']), idx, 0)
                self.slot_layout.addWidget(combo, idx, 1)

        except Exception as e:
            self._log(f"Error loading map: {str(e)}")
            traceback.print_exc()

    def _log(self, msg):
        self.log_box.appendPlainText(msg)

    def _update_preview(self):
        current_idx = self.result_combo.currentIndex()
        if 0 <= current_idx < len(self.result_images):
            image_path = self.result_images[current_idx]
            if os.path.exists(image_path):
                pixmap = QPixmap(image_path)
                scaled_pixmap = pixmap.scaledToWidth(500, Qt.TransformationMode.SmoothTransformation)
                self.image_label.setPixmap(scaled_pixmap)

    def _apply_gui_params(self):
        for key, widget in self.param_inputs.items():
            if key == 'max_search_steps':
                PARAMS[key] = int(widget.value())
            else:
                PARAMS[key] = float(widget.value())

    def run_planning(self):
        try:
            self.run_btn.setEnabled(False)
            self.log_box.clear()
            self.result_images = []
            self.result_combo.clear()
            self.image_label.setPixmap(QPixmap())
            QApplication.processEvents()

            self._apply_gui_params()

            output_dir = self.output_dir_input.text().strip()
            if not output_dir:
                raise ValueError('Output folder cannot be empty.')
            os.makedirs(output_dir, exist_ok=True)

            start = State(
                float(self.start_x_input.value()),
                float(self.start_y_input.value()),
                float(self.start_theta_input.value()),
            )
            
            obs_map = self.current_obs_map
            
            scenarios = []
            spots = self.current_scenarios_config.get('spots', [])
            for spot in spots:
                name = spot['name']
                gx = spot['center_x']
                gy = float(self.slot_inputs[name].currentData())
                outward_theta = spot['outward_theta']

                wb = PARAMS['wheelbase']
                out_goal_x = gx - (wb / 2.0) * math.cos(outward_theta)
                out_goal_y = gy - (wb / 2.0) * math.sin(outward_theta)
                scenarios.append((name, 'outward', State(out_goal_x, out_goal_y, outward_theta)))

            total_scenarios = len(scenarios)
            mode_name = str(self.mode_combo.currentData())
            planning_func = multi_stage_planning if mode_name == 'classic' else multi_stage_planning_directional
            self._log(f'Planner mode: {mode_name}')

            success_count = 0
            successful_planning_times = []
            run_start_t = time.perf_counter()
            for idx, (quadrant, orientation, goal) in enumerate(scenarios, start=1):
                scenario_start_t = time.perf_counter()
                self._log(f'===== [{idx}/{total_scenarios}] {quadrant} - {orientation} =====')
                QApplication.processEvents()

                final_node, all_explored = None, []
                used_strategy = 'none'
                for strategy_name, updates in PARAM_STRATEGIES:
                    apply_param_strategy(strategy_name, updates)
                    self._log(f'Use strategy: {strategy_name}')
                    QApplication.processEvents()

                    final_node, all_explored = planning_func(start, goal, obs_map)
                    if final_node:
                        used_strategy = strategy_name
                        break

                if not final_node:
                    scenario_elapsed = time.perf_counter() - scenario_start_t
                    self._log(f'  [Time] 規劃耗時: {scenario_elapsed:.3f} 秒')
                    self._log('  [Failed] 找不到可行路徑')
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
                self.result_images.append(image_path)
                success_count += 1
                scenario_elapsed = time.perf_counter() - scenario_start_t
                successful_planning_times.append(scenario_elapsed)
                self._log(f'  [Saved] {image_path}')
                self._log(f'  [Time] 規劃耗時: {scenario_elapsed:.3f} 秒')
                QApplication.processEvents()

            # Update combo box with results
            for i, path in enumerate(self.result_images, 1):
                display_name = os.path.basename(path)
                self.result_combo.addItem(display_name, path)

            # Display the first image if available
            if self.result_images:
                self.result_combo.setCurrentIndex(0)

            total_elapsed = time.perf_counter() - run_start_t
            if successful_planning_times:
                avg_time = sum(successful_planning_times) / len(successful_planning_times)
                avg_line = f'Average successful path time: {avg_time:.3f} s ({success_count} paths)'
            else:
                avg_line = 'Average successful path time: N/A (0 paths)'

            self._log(f'\nDone. Success: {success_count}/{total_scenarios}')
            self._log(avg_line)
            self._log(f'Total elapsed time: {total_elapsed:.3f} s')

            QMessageBox.information(
                self,
                'Done',
                f'Planning completed. Success: {success_count}/{total_scenarios}\n{avg_line}\nTotal elapsed: {total_elapsed:.3f} s',
            )
        except Exception as exc:
            self._log('ERROR:\n' + traceback.format_exc())
            QMessageBox.critical(self, 'Error', str(exc))
        finally:
            self.run_btn.setEnabled(True)


if __name__ == '__main__':
    app = QApplication([])
    app.setApplicationName('Hybrid A* GUI')
    win = PlannerGui()
    win.show()
    app.exec()
