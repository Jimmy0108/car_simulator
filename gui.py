import math
import os
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
    AISLE_RIGHT_CENTER_X,
    DEFAULT_QUADRANT_SLOT_INDEX,
    PARAMS,
    PARAM_STRATEGIES,
    QUADRANT_SLOT_OPTIONS,
    apply_param_strategy,
)
from models import State
from parking_map import UndergroundParkingMap
from planner import multi_stage_planning
from scenarios import get_quadrant_parking_scenarios
from visualization import plot_results


class PlannerGui(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Hybrid A* Parameter Console')
        self.resize(1400, 900)

        self.param_inputs = {}
        self.slot_inputs = {}
        self.result_images = []  # Store paths of generated images

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
            'w1',
            'w2',
            'w3',
            'u1',
            'u2',
            'u3',
            'u4',
            'alpha',
            'd_o_max',
            'vehicle_L',
            'vehicle_W',
            'wheelbase',
            'turning_radius',
            'max_steer_angle',
            'reverse_penalty',
            'switch_gear_penalty',
            'steerr_penalty',
            'steer_change_penalty',
            'center_clearance_buffer',
            'corner_clearance',
            'step_size',
            'max_search_steps',
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

        start_form = QFormLayout()
        self.start_x_input = QDoubleSpinBox()
        self.start_x_input.setRange(-1000.0, 1000.0)
        self.start_x_input.setDecimals(3)
        self.start_x_input.setValue(float(AISLE_RIGHT_CENTER_X))

        self.start_y_input = QDoubleSpinBox()
        self.start_y_input.setRange(-1000.0, 1000.0)
        self.start_y_input.setDecimals(3)
        self.start_y_input.setValue(44.0)

        self.start_theta_input = QDoubleSpinBox()
        self.start_theta_input.setRange(-6.3, 6.3)
        self.start_theta_input.setDecimals(4)
        self.start_theta_input.setValue(-math.pi / 2)

        start_form.addRow('Start X', self.start_x_input)
        start_form.addRow('Start Y', self.start_y_input)
        start_form.addRow('Start Theta(rad)', self.start_theta_input)
        layout.addLayout(start_form)

        slot_group = QGroupBox('Quadrant Slots')
        slot_layout = QGridLayout(slot_group)
        ordered_quadrants = [
            'Q1_upper_left',
            'Q2_upper_right',
            'Q3_lower_left',
            'Q4_lower_right',
        ]

        for idx, quadrant in enumerate(ordered_quadrants):
            combo = QComboBox()
            options = QUADRANT_SLOT_OPTIONS[quadrant]
            for y in options:
                combo.addItem(f'y={y}', y)
            default_idx = DEFAULT_QUADRANT_SLOT_INDEX[quadrant] - 1
            combo.setCurrentIndex(max(0, min(default_idx, combo.count() - 1)))
            self.slot_inputs[quadrant] = combo
            slot_layout.addWidget(QLabel(quadrant), idx, 0)
            slot_layout.addWidget(combo, idx, 1)

        layout.addWidget(slot_group)

        strategy_group = QGroupBox('Strategy Order')
        strategy_layout = QVBoxLayout(strategy_group)
        self.strategy_label = QLabel(' -> '.join([name for name, _ in PARAM_STRATEGIES]))
        self.strategy_label.setWordWrap(True)
        strategy_layout.addWidget(self.strategy_label)
        layout.addWidget(strategy_group)

        return group

    def _log(self, msg):
        self.log_box.appendPlainText(msg)

    def _update_preview(self):
        """Update the image preview when selection changes"""
        current_idx = self.result_combo.currentIndex()
        if 0 <= current_idx < len(self.result_images):
            image_path = self.result_images[current_idx]
            if os.path.exists(image_path):
                pixmap = QPixmap(image_path)
                # Scale to fit in preview area while maintaining aspect ratio
                scaled_pixmap = pixmap.scaledToWidth(500, Qt.TransformationMode.SmoothTransformation)
                self.image_label.setPixmap(scaled_pixmap)

    def _apply_gui_params(self):
        for key, widget in self.param_inputs.items():
            if key == 'max_search_steps':
                PARAMS[key] = int(widget.value())
            else:
                PARAMS[key] = float(widget.value())

    def _collect_selected_slots(self):
        return {q: float(combo.currentData()) for q, combo in self.slot_inputs.items()}

    def run_planning(self):
        try:
            self.run_btn.setEnabled(False)
            self.log_box.clear()
            self.result_images = []
            self.result_combo.clear()
            self.image_label.setPixmap(QPixmap())
            QApplication.processEvents()

            self._apply_gui_params()
            selected_slots = self._collect_selected_slots()

            output_dir = self.output_dir_input.text().strip()
            if not output_dir:
                raise ValueError('Output folder cannot be empty.')
            os.makedirs(output_dir, exist_ok=True)

            start = State(
                float(self.start_x_input.value()),
                float(self.start_y_input.value()),
                float(self.start_theta_input.value()),
            )
            obs_map = UndergroundParkingMap()
            scenarios = get_quadrant_parking_scenarios(selected_slots)

            success_count = 0
            for idx, (quadrant, orientation, goal) in enumerate(scenarios, start=1):
                self._log(f'===== [{idx}/8] {quadrant} - {orientation} =====')
                QApplication.processEvents()

                final_node, all_explored = None, []
                used_strategy = 'none'
                for strategy_name, updates in PARAM_STRATEGIES:
                    apply_param_strategy(strategy_name, updates)
                    self._log(f'Use strategy: {strategy_name}')
                    QApplication.processEvents()

                    final_node, all_explored = multi_stage_planning(start, goal, obs_map)
                    if final_node:
                        used_strategy = strategy_name
                        break

                if not final_node:
                    self._log('  [Failed] 找不到可行路徑')
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
                self.result_images.append(image_path)
                success_count += 1
                self._log(f'  [Saved] {image_path}')
                QApplication.processEvents()

            # Update combo box with results
            for i, path in enumerate(self.result_images, 1):
                display_name = os.path.basename(path)
                self.result_combo.addItem(display_name, path)

            # Display the first image if available
            if self.result_images:
                self.result_combo.setCurrentIndex(0)

            self._log(f'\nDone. Success: {success_count}/8')
            QMessageBox.information(self, 'Done', f'Planning completed. Success: {success_count}/8')
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
