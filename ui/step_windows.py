"""
Step windows module - unified font sizes and adaptive windows.

Uses Step2Window as the reference: large fonts throughout, buttons size to
their content, and windows adjust to their content. Contains all original
feature logic without omissions.
"""

import time
import traceback
import numpy as np
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QRadioButton,
                             QButtonGroup, QFileDialog, QMessageBox,
                             QSpinBox, QGroupBox, QComboBox, QDoubleSpinBox)
from .result_windows import ImageClusterWindow, ColorInputWindow, Step4Window
from .widgets import DynamicResultsWindow, StaticResultsWindow, ColorConverter
from .window_utils import apply_main_geometry
from .worker import run_with_progress
from utils.helpers import safe_flush_stdout, get_default_dir, get_image_dir


class Step1Window(QWidget):
    """Step 1: input selection window - unified fonts, window adapts to content"""
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.input_type = None
        self.image_type = None  # 'single' or 'multiple'
        self.image_paths = []  # Supports multiple images
        self.color_groups = []
        self.step2_window = None
        self.cluster_window = None
        self.color_window = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Step 1: Input Selection")

        # Main layout: vertical, large spacing, wide margins
        layout = QVBoxLayout()
        layout.setSpacing(30)
        layout.setContentsMargins(60, 40, 60, 40)

        # === Input type selection (two large radio buttons) ===
        input_group = QButtonGroup(self)
        self.image_radio = QRadioButton("Image Input")
        self.color_radio = QRadioButton("Color Input")
        # 32px font with extra padding
        radio_style = "font-size: 32px; padding: 12px;"
        self.image_radio.setStyleSheet(radio_style)
        self.color_radio.setStyleSheet(radio_style)
        input_group.addButton(self.image_radio)
        input_group.addButton(self.color_radio)
        layout.addWidget(self.image_radio)
        layout.addWidget(self.color_radio)

        # === Image type selection (only shown for image input) ===
        self.image_type_group = QWidget()
        image_type_layout = QVBoxLayout()
        image_type_layout.setContentsMargins(40, 10, 0, 10)
        image_type_layout.setSpacing(15)

        image_type_button_group = QButtonGroup(self)
        self.single_image_radio = QRadioButton("Single Image")
        self.multiple_images_radio = QRadioButton("Multiple Images")
        # Secondary radio buttons use 28px font
        sub_radio_style = "font-size: 28px; padding: 8px;"
        self.single_image_radio.setStyleSheet(sub_radio_style)
        self.multiple_images_radio.setStyleSheet(sub_radio_style)
        self.single_image_radio.setChecked(True)
        image_type_button_group.addButton(self.single_image_radio)
        image_type_button_group.addButton(self.multiple_images_radio)
        image_type_layout.addWidget(self.single_image_radio)
        image_type_layout.addWidget(self.multiple_images_radio)
        self.image_type_group.setLayout(image_type_layout)
        self.image_type_group.setVisible(False)
        layout.addWidget(self.image_type_group)

        # === Single-image selection area ===
        self.single_image_group = QWidget()
        single_image_layout = QVBoxLayout()
        single_image_layout.setContentsMargins(40, 10, 0, 10)
        single_image_layout.setSpacing(15)

        self.single_image_label = QLabel("No image selected")
        self.single_image_label.setStyleSheet("color: #888; font-size: 24px;")
        single_image_layout.addWidget(self.single_image_label)

        self.select_single_btn = QPushButton("Select Image")
        self.select_single_btn.setStyleSheet("font-size: 28px; padding: 12px;")
        self.select_single_btn.clicked.connect(self.select_single_image)
        single_image_layout.addWidget(self.select_single_btn)

        self.single_image_group.setLayout(single_image_layout)
        self.single_image_group.setVisible(False)
        layout.addWidget(self.single_image_group)

        # === Multiple-image selection area ===
        self.multiple_images_group = QWidget()
        multiple_images_layout = QVBoxLayout()
        multiple_images_layout.setContentsMargins(40, 10, 0, 10)
        multiple_images_layout.setSpacing(15)

        # Image count selection
        count_layout = QHBoxLayout()
        count_label = QLabel("Number of images:")
        count_label.setStyleSheet("font-size: 24px;")
        self.image_count_spin = QSpinBox()
        self.image_count_spin.setRange(2, 10)
        self.image_count_spin.setValue(3)
        self.image_count_spin.valueChanged.connect(self.update_image_slots)
        self.image_count_spin.setStyleSheet("font-size: 24px; padding: 8px;")
        count_layout.addWidget(count_label)
        count_layout.addWidget(self.image_count_spin)
        count_layout.addStretch()
        multiple_images_layout.addLayout(count_layout)

        # Image list container
        self.image_slots_layout = QVBoxLayout()
        self.image_slots_layout.setSpacing(10)
        self.image_slots = []  # Stores the image selection widgets
        multiple_images_layout.addLayout(self.image_slots_layout)

        self.multiple_images_group.setLayout(multiple_images_layout)
        self.multiple_images_group.setVisible(False)
        layout.addWidget(self.multiple_images_group)

        # === Next button ===
        self.next_btn = QPushButton("Next")
        self.next_btn.setStyleSheet("""
            QPushButton {
                font-size: 28px;
                padding: 16px;
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        self.next_btn.clicked.connect(self.next_step)
        layout.addWidget(self.next_btn)

        # Connect signals
        self.image_radio.toggled.connect(self.toggle_image_type_options)
        self.single_image_radio.toggled.connect(self.toggle_image_input_method)
        self.multiple_images_radio.toggled.connect(self.toggle_image_input_method)

        # Initialize image slots
        self.update_image_slots(3)

        layout.addStretch()
        self.setLayout(layout)

        # Apply unified window size and center
        self.adjustSize()
        apply_main_geometry(self)

    def toggle_image_type_options(self, checked):
        """Toggle visibility of the image type options"""
        self.image_type_group.setVisible(checked)
        if checked:
            self.toggle_image_input_method()
        else:
            self.single_image_group.setVisible(False)
            self.multiple_images_group.setVisible(False)
        self.adjustSize()

    def toggle_image_input_method(self):
        """Switch between single/multiple image input modes"""
        if self.single_image_radio.isChecked():
            self.single_image_group.setVisible(True)
            self.multiple_images_group.setVisible(False)
        else:
            self.single_image_group.setVisible(False)
            self.multiple_images_group.setVisible(True)
        self.adjustSize()

    def update_image_slots(self, count):
        """Update the image selection slots"""
        for slot in self.image_slots:
            if slot['widget'].layout():
                while slot['widget'].layout().count():
                    child = slot['widget'].layout().takeAt(0)
                    if child.widget():
                        child.widget().deleteLater()
                slot['widget'].deleteLater()
        self.image_slots.clear()

        for i in range(count):
            slot_widget = self.create_image_slot_widget(i + 1)
            self.image_slots.append(slot_widget)
            self.image_slots_layout.addWidget(slot_widget['widget'])

        self.adjustSize()

    def create_image_slot_widget(self, index):
        """Create a single image selection slot widget"""
        widget = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(15)

        label = QLabel(f"Image {index}:")
        label.setFixedWidth(120)
        label.setStyleSheet("font-size: 24px;")
        layout.addWidget(label)

        path_label = QLabel("Not selected")
        path_label.setStyleSheet("color: #888; font-size: 22px;")
        path_label.setFixedWidth(400)
        layout.addWidget(path_label)

        select_btn = QPushButton("Select")
        select_btn.setStyleSheet("font-size: 22px; padding: 8px;")
        select_btn.clicked.connect(lambda checked, idx=index-1, lbl=path_label: self.select_image_for_slot(idx, lbl))
        layout.addWidget(select_btn)

        clear_btn = QPushButton("Clear")
        clear_btn.setStyleSheet("font-size: 22px; padding: 8px;")
        clear_btn.clicked.connect(lambda checked, idx=index-1, lbl=path_label: self.clear_image_slot(idx, lbl))
        layout.addWidget(clear_btn)

        layout.addStretch()
        widget.setLayout(layout)

        return {
            'widget': widget,
            'path_label': path_label,
            'select_btn': select_btn,
            'clear_btn': clear_btn,
            'image_path': None
        }

    def select_single_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", get_image_dir(), "Image Files (*.png *.jpg *.bmp *.jpeg)"
        )
        if file_path:
            self.single_image_label.setText(f"Selected: {file_path}")
            self.single_image_label.setStyleSheet("color: #333; font-size: 24px;")
            self.image_paths = [file_path]
            self.adjustSize()

    def select_image_for_slot(self, slot_index, label_widget):
        file_path, _ = QFileDialog.getOpenFileName(
            self, f"Select Image {slot_index + 1}", get_image_dir(), "Image Files (*.png *.jpg *.bmp *.jpeg)"
        )
        if file_path:
            display = f"...{file_path[-30:]}" if len(file_path) > 30 else file_path
            label_widget.setText(display)
            label_widget.setStyleSheet("color: #333; font-size: 22px;")
            label_widget.setToolTip(file_path)
            self.image_slots[slot_index]['image_path'] = file_path
            self.adjustSize()

    def clear_image_slot(self, slot_index, label_widget):
        label_widget.setText("Not selected")
        label_widget.setStyleSheet("color: #888; font-size: 22px;")
        label_widget.setToolTip("")
        self.image_slots[slot_index]['image_path'] = None
        self.adjustSize()

    def next_step(self):
        if self.image_radio.isChecked():
            self.input_type = 'image'
            if self.single_image_radio.isChecked():
                self.image_type = 'single'
                if not self.image_paths:
                    QMessageBox.warning(self, "Warning", "Please select an image first")
                    return
                def work(report, worker):
                    report(-1, "Extracting dominant colors from image...")
                    return self.extract_dominant_colors(self.image_paths[0])

                def on_success(colors):
                    self.cluster_window = ImageClusterWindow(colors, parent=self)
                    self.cluster_window.show()
                    self.open_step2()

                def on_error(err):
                    QMessageBox.critical(self, "Error", f"Color extraction failed: {err}")

                run_with_progress(self, "Color Extraction",
                                  "Extracting dominant colors from image...",
                                  work, on_success, on_error)
            elif self.multiple_images_radio.isChecked():
                self.image_type = 'multiple'
                selected_paths = []
                missing_slots = []
                for i, slot in enumerate(self.image_slots):
                    if slot['image_path']:
                        selected_paths.append(slot['image_path'])
                    else:
                        missing_slots.append(i + 1)

                if not selected_paths:
                    QMessageBox.warning(self, "Warning", "Please select at least one image")
                    return

                if missing_slots:
                    reply = QMessageBox.question(
                        self, "Confirm",
                        f"Images {', '.join(map(str, missing_slots))} not selected. Continue with selected images?",
                        QMessageBox.Yes | QMessageBox.No
                    )
                    if reply == QMessageBox.No:
                        return

                self.image_paths = selected_paths

                def work(report, worker):
                    report(-1, f"Extracting dominant colors from {len(selected_paths)} images...")
                    colors = self.extract_dominant_colors(selected_paths)
                    report(-1, "Removing similar colors...")
                    return self.remove_similar_colors(list(colors), threshold=30)

                def on_success(unique_colors):
                    if not unique_colors:
                        QMessageBox.warning(self, "Warning", "No valid colors extracted")
                        return
                    self.cluster_window = ImageClusterWindow(
                        unique_colors,
                        title="Dominant Colors from Multiple Images",
                        parent=self
                    )
                    self.cluster_window.show()
                    self.open_step2()

                def on_error(err):
                    QMessageBox.critical(self, "Error", f"Color extraction failed: {err}")

                run_with_progress(self, "Color Extraction",
                                  "Extracting dominant colors...",
                                  work, on_success, on_error)

        elif self.color_radio.isChecked():
            self.input_type = 'color'
            self.color_window = ColorInputWindow(parent=self)
            self.color_window.show()

    def remove_similar_colors(self, colors, threshold=30):
        if not colors:
            return []
        unique_colors = []
        for color in colors:
            is_similar = False
            for unique_color in unique_colors:
                distance = ((color[0] - unique_color[0]) ** 2 +
                            (color[1] - unique_color[1]) ** 2 +
                            (color[2] - unique_color[2]) ** 2) ** 0.5
                if distance < threshold:
                    is_similar = True
                    break
            if not is_similar:
                unique_colors.append(color)
        return unique_colors[:15]

    def extract_dominant_colors(self, image_path, n_colors=None):
        from core.color_processing import extract_dominant_colors
        try:
            colors, weights = extract_dominant_colors(image_path, n_colors)
            # Save colors and weights to color_groups
            self.color_groups = [{'space': 'RGB', 'values': list(color), 'weight': weight} for color, weight in
                                 zip(colors, weights)]
            return colors
        except ImportError as e:
            raise ImportError(f"Cannot import color processing module: {e}") from e
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Image file not found: {image_path}") from e
        except ValueError as e:
            raise ValueError(f"Image processing error: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Color extraction failed: {e}. Image path: {image_path}") from e

    def color_input_confirmed(self, color_groups):
        self.color_groups = color_groups
        self.open_step2()

    def open_step2(self):
        self.hide()
        self.step2_window = Step2Window(self)
        self.step2_window.show()


class Step2Window(QWidget):
    """Step 2: design type selection - already meets requirements, kept unchanged"""
    def __init__(self, prev_window):
        super().__init__()
        self.prev_window = prev_window
        self.design_type = None  # 'dynamic' or 'static'
        self.color_groups = prev_window.color_groups
        self.color_count = len(self.color_groups)
        self.step3_window = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Step 2: Design Type")

        layout = QVBoxLayout()
        layout.setSpacing(30)
        layout.setContentsMargins(60, 40, 60, 40)

        # Show the current input type
        if self.prev_window.input_type == 'image':
            if self.prev_window.image_type == 'single':
                input_text = "Input: Single Image"
            else:
                input_text = f"Input: {len(self.prev_window.image_paths)} Images"
        else:
            input_text = f"Input: {self.color_count} Color Groups"
        input_label = QLabel(input_text)
        input_label.setStyleSheet("font-size: 28px; color: #555; font-weight: bold;")
        input_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(input_label)

        # Design type selection
        design_group = QButtonGroup(self)
        self.dynamic_radio = QRadioButton("Dynamic Design")
        self.static_radio = QRadioButton("Static Design")
        radio_style = "font-size: 32px; padding: 12px;"
        self.dynamic_radio.setStyleSheet(radio_style)
        self.static_radio.setStyleSheet(radio_style)
        design_group.addButton(self.dynamic_radio)
        design_group.addButton(self.static_radio)
        layout.addWidget(self.dynamic_radio)
        layout.addWidget(self.static_radio)

        # Next button
        next_btn = QPushButton("Next")
        next_btn.setStyleSheet("""
            QPushButton {
                font-size: 28px;
                padding: 16px;
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        next_btn.clicked.connect(self.next_step)
        layout.addWidget(next_btn)

        # Connect signals
        self.dynamic_radio.toggled.connect(self.set_design_type)
        self.static_radio.toggled.connect(self.set_design_type)

        layout.addStretch()
        self.setLayout(layout)
        self.adjustSize()
        apply_main_geometry(self)

    def set_design_type(self):
        if self.dynamic_radio.isChecked():
            self.design_type = 'dynamic'
        elif self.static_radio.isChecked():
            self.design_type = 'static'

    def next_step(self):
        if self.design_type is None:
            QMessageBox.warning(self, "Warning", "Please select a design type")
            return
        self.hide()
        self.step3_window = Step3Window(self)
        self.step3_window.show()


class Step3Window(QWidget):
    """Step 3: parameter setup and generation - unified fonts, adaptive window"""
    def __init__(self, prev_window):
        super().__init__()
        self.prev_window = prev_window
        self.design_type = prev_window.design_type
        self.color_groups = prev_window.color_groups
        self.color_count = len(self.color_groups)
        self.num_samples = 200 if self.design_type == 'dynamic' else 500
        self.deltaE_threshold = 10.0 if self.design_type == 'dynamic' else 5.0
        self.clustering_method = 'dbscan' if self.design_type == 'dynamic' else None
        self.selected_designs = []
        self.all_design_results = []
        self.all_solutions = []
        self.init_ui()
        self.update_status()

    def init_ui(self):
        self.setWindowTitle("Step 3: Parameters & Generation")

        self.main_layout = QVBoxLayout()
        self.main_layout.setSpacing(25)
        self.main_layout.setContentsMargins(60, 40, 60, 40)

        # Status info (enlarged font)
        self.status_label = QLabel()
        self.status_label.setStyleSheet("font-size: 28px; color: #555; font-weight: bold; padding: 8px; background-color: #f0f0f0; border-radius: 5px;")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.status_label)

        # Parameter group
        params_group = QGroupBox("Parameters")
        params_group.setStyleSheet("font-size: 24px; font-weight: bold;")
        params_layout = QVBoxLayout()
        params_layout.setSpacing(20)

        # Number of samples
        samples_layout = QHBoxLayout()
        samples_label = QLabel("Number of samples:")
        samples_label.setStyleSheet("font-size: 24px;")
        self.samples_spin = QSpinBox()
        if self.design_type == 'dynamic':
            self.samples_spin.setRange(100, 10000)
            self.samples_spin.setValue(self.num_samples)
        else:
            self.samples_spin.setRange(1000, 20000)
            self.samples_spin.setValue(self.num_samples)
        self.samples_spin.setStyleSheet("font-size: 24px; padding: 8px;")
        self.samples_spin.valueChanged.connect(self.update_samples)
        samples_layout.addWidget(samples_label)
        samples_layout.addWidget(self.samples_spin)
        samples_layout.addStretch()
        params_layout.addLayout(samples_layout)

        # ΔE threshold
        deltae_layout = QHBoxLayout()
        deltae_label = QLabel("ΔE threshold:")
        deltae_label.setStyleSheet("font-size: 24px;")
        self.deltae_spin = QDoubleSpinBox()
        if self.design_type == 'dynamic':
            self.deltae_spin.setRange(0.0, 100.0)
            self.deltae_spin.setValue(self.deltaE_threshold)
            self.deltae_spin.setSingleStep(0.5)
        else:
            self.deltae_spin.setRange(0.0, 50.0)
            self.deltae_spin.setValue(self.deltaE_threshold)
            self.deltae_spin.setSingleStep(0.1)
        self.deltae_spin.setStyleSheet("font-size: 24px; padding: 8px;")
        self.deltae_spin.valueChanged.connect(self.update_deltae)
        deltae_layout.addWidget(deltae_label)
        deltae_layout.addWidget(self.deltae_spin)
        deltae_layout.addStretch()
        params_layout.addLayout(deltae_layout)

        # Clustering method (dynamic only)
        if self.design_type == 'dynamic':
            clustering_layout = QHBoxLayout()
            clustering_label = QLabel("Clustering method:")
            clustering_label.setStyleSheet("font-size: 24px;")
            self.clustering_combo = QComboBox()
            self.clustering_combo.addItems(["kmeans_auto", "dbscan"])
            self.clustering_combo.setCurrentText(self.clustering_method)
            self.clustering_combo.currentTextChanged.connect(self.update_clustering_method)
            self.clustering_combo.setStyleSheet("font-size: 24px; padding: 8px;")
            clustering_layout.addWidget(clustering_label)
            clustering_layout.addWidget(self.clustering_combo)
            clustering_layout.addStretch()
            params_layout.addLayout(clustering_layout)

        params_group.setLayout(params_layout)
        self.main_layout.addWidget(params_group)

        # Generate button
        self.generate_btn = QPushButton("Generate Designs")
        self.generate_btn.setStyleSheet("""
            QPushButton {
                font-size: 28px;
                padding: 16px;
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        self.generate_btn.clicked.connect(self.generate_designs)
        self.main_layout.addWidget(self.generate_btn)

        # Progress label
        self.progress_label = QLabel("")
        self.progress_label.setStyleSheet("color: #0066cc; font-size: 20px;")
        self.progress_label.setVisible(False)
        self.main_layout.addWidget(self.progress_label)

        self.main_layout.addStretch()
        self.setLayout(self.main_layout)

        self.adjustSize()
        apply_main_geometry(self)

    def update_status(self):
        status_text = f"Colors: {self.color_count} | Type: {'Dynamic' if self.design_type == 'dynamic' else 'Static'}"
        self.status_label.setText(status_text)

    def update_samples(self, value):
        self.num_samples = value

    def update_deltae(self, value):
        self.deltaE_threshold = value

    def update_clustering_method(self, method):
        self.clustering_method = method

    def generate_designs(self):
        """Run design generation in a background thread while the main thread shows a progress dialog"""
        colors = self.color_groups
        if not colors:
            self.show_error("No valid color data")
            return

        self.generate_btn.setEnabled(False)
        self.generate_btn.setText("Generating...")
        self.progress_label.setText("Processing...")
        self.progress_label.setVisible(True)

        self._design_start = time.perf_counter()
        total = len(colors)
        design_type = self.design_type

        def work(report, worker):
            results = []
            for i, color_group in enumerate(colors):
                worker.check_cancel()
                report(int(i * 100 / total), f"Processing color {i + 1}/{total}...")
                target_rgb = color_group['values']
                try:
                    if design_type == 'dynamic':
                        result = self.generate_dynamic_design_single(target_rgb)
                    else:
                        result = self.generate_static_design_single(target_rgb)

                    if result:
                        result['color_index'] = i
                        result['target_rgb'] = target_rgb
                        results.append(result)
                except Exception as e:
                    print(f"Color {i + 1} failed: {e}")
                    continue
            report(100, "Finalizing...")
            return results

        def on_success(results):
            self.design_elapsed = time.perf_counter() - self._design_start
            print(f"Design generation took {self.design_elapsed:.1f} s")
            self.all_design_results = results
            self.selected_designs = []

            if not results:
                self.show_error("No design results generated")
                self.reset_generate_btn()
                return

            if self.design_type == 'dynamic':
                self.process_dynamic_results()
            else:
                self.process_static_results()

        def on_error(err):
            print(f"Generation failed: {err}")
            self.show_error(f"Generation failed: {err}")
            self.reset_generate_btn()

        def on_cancel():
            print("Generation cancelled by user")
            self.reset_generate_btn()

        run_with_progress(self, "Generating Designs",
                          "Starting generation...",
                          work, on_success, on_error, on_cancel,
                          cancellable=True)

    def reset_generate_btn(self):
        self.generate_btn.setEnabled(True)
        self.generate_btn.setText("Generate Designs")
        self.progress_label.setVisible(False)

    def generate_dynamic_design_single(self, target_rgb):
        try:
            from core.design_generation import design_and_cluster_by_color
            result = design_and_cluster_by_color(
                target_rgb=target_rgb,
                num_samples=self.num_samples,
                clustering_method=self.clustering_method,
                deltaE_threshold=self.deltaE_threshold
            )
            if result and 'solutions' in result:
                solutions = result['solutions']
                if 'cluster_best' not in solutions:
                    solutions['cluster_best'] = []
                elif solutions['cluster_best'] is None:
                    solutions['cluster_best'] = []
                if 'global_best' not in solutions:
                    solutions['global_best'] = None
                elif isinstance(solutions['global_best'], str):
                    solutions['global_best'] = None
                if 'max_deltaED' not in solutions:
                    solutions['max_deltaED'] = None
                elif isinstance(solutions['max_deltaED'], str):
                    solutions['max_deltaED'] = None

                for field in ['cluster_best', 'global_best', 'max_deltaED']:
                    if field == 'cluster_best':
                        for i, solution in enumerate(solutions[field]):
                            if isinstance(solution, dict):
                                if 'thickness' not in solution:
                                    solution['thickness'] = []
                                if 'deltaE' not in solution:
                                    solution['deltaE'] = 0
                                if 'deltaED' not in solution:
                                    solution['deltaED'] = 0
                                if 'pred_lab_amorphous' not in solution:
                                    solution['pred_lab_amorphous'] = [50, 0, 0]
                                if 'pred_lab_crystalline' not in solution:
                                    solution['pred_lab_crystalline'] = [50, 0, 0]
                                if 'cluster_id' not in solution:
                                    solution['cluster_id'] = i
                    elif solutions[field] and isinstance(solutions[field], dict):
                        solution = solutions[field]
                        if 'thickness' not in solution:
                            solution['thickness'] = []
                        if 'deltaE' not in solution:
                            solution['deltaE'] = 0
                        if 'deltaED' not in solution:
                            solution['deltaED'] = 0
                        if 'pred_lab_amorphous' not in solution:
                            solution['pred_lab_amorphous'] = [50, 0, 0]
                        if 'pred_lab_crystalline' not in solution:
                            solution['pred_lab_crystalline'] = [50, 0, 0]
            return result
        except ImportError as e:
            print(f"Dynamic design module import failed: {e}")
            raise
        except Exception as e:
            print(f"Dynamic design generation failed: {e}")
            print(f"traceback: {traceback.format_exc()}")
            raise

    def generate_static_design_single(self, target_rgb):
        try:
            from core.design_generation import generate_accurate_design
            result = generate_accurate_design(
                target_rgb=target_rgb,
                num_samples=self.num_samples,
                deltaE_threshold=self.deltaE_threshold
            )
            if result and 'solutions' in result:
                solutions = result['solutions']
                if 'global_best' not in solutions:
                    solutions['global_best'] = None
                elif isinstance(solutions['global_best'], str):
                    solutions['global_best'] = None
                if solutions['global_best'] and isinstance(solutions['global_best'], dict):
                    global_best = solutions['global_best']
                    if 'thickness' not in global_best:
                        global_best['thickness'] = []
                    if 'deltaE' not in global_best:
                        global_best['deltaE'] = 0
                    if 'deltaED' not in global_best:
                        global_best['deltaED'] = 0
                    if 'pred_lab_amorphous' not in global_best:
                        global_best['pred_lab_amorphous'] = [50, 0, 0]
                    if 'pred_lab_crystalline' not in global_best:
                        global_best['pred_lab_crystalline'] = [50, 0, 0]
            return result
        except ImportError as e:
            print(f"Static design module import failed: {e}")
            raise
        except Exception as e:
            print(f"Static design generation failed: {e}")
            print(f"traceback: {traceback.format_exc()}")
            raise

    def process_dynamic_results(self):
        try:
            self.all_solutions = []
            for design_result in self.all_design_results:
                color_index = design_result['color_index']
                target_rgb = design_result['target_rgb']
                solutions_data = design_result['solutions']

                if 'cluster_best' in solutions_data and solutions_data['cluster_best']:
                    for cluster_solution in solutions_data['cluster_best']:
                        if cluster_solution and isinstance(cluster_solution, dict):
                            solution = self.create_solution_dict(
                                color_index=color_index,
                                solution_type='cluster_best',
                                solution_data=cluster_solution,
                                target_rgb=target_rgb,
                                cluster_id=cluster_solution.get('cluster_id', -1)
                            )
                            self.all_solutions.append(solution)

                if 'global_best' in solutions_data and solutions_data['global_best']:
                    global_best = solutions_data['global_best']
                    if isinstance(global_best, dict):
                        solution = self.create_solution_dict(
                            color_index=color_index,
                            solution_type='global_best',
                            solution_data=global_best,
                            target_rgb=target_rgb,
                            cluster_id=-1
                        )
                        self.all_solutions.append(solution)

                if 'max_deltaED' in solutions_data and solutions_data['max_deltaED']:
                    max_deltaED = solutions_data['max_deltaED']
                    if isinstance(max_deltaED, dict):
                        solution = self.create_solution_dict(
                            color_index=color_index,
                            solution_type='max_deltaED',
                            solution_data=max_deltaED,
                            target_rgb=target_rgb,
                            cluster_id=-2
                        )
                        self.all_solutions.append(solution)

            if self.all_solutions:
                self.dynamic_results_window = DynamicResultsWindow(self.all_solutions, parent=self)
                self.dynamic_results_window.show()
                self.hide()
            else:
                self.show_error("No solutions generated")
                self.reset_generate_btn()

        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            with open('debug.log', 'a', encoding='utf-8') as f:
                f.write(f"process_dynamic_results error:\n{error_msg}\n")
            print(f"process_dynamic_results error: {e}")
            try:
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.critical(self, "Error", f"Failed to process dynamic results:\n{str(e)}")
            except:
                pass
            self.reset_generate_btn()

    def process_static_results(self):
        try:
            simplified_designs = []
            for design_result in self.all_design_results:
                color_index = design_result['color_index']
                target_rgb = design_result['target_rgb']
                solutions_data = design_result['solutions']
                global_best = solutions_data['global_best']
                if global_best:
                    amorphous_lab = global_best.get('pred_lab_amorphous', [50, 0, 0])
                    crystalline_lab = global_best.get('pred_lab_crystalline', [50, 0, 0])
                    amorphous_rgb = ColorConverter.lab_to_rgb(amorphous_lab)
                    crystalline_rgb = ColorConverter.lab_to_rgb(crystalline_lab)
                    color_design = {
                        'color_index': color_index,
                        'target_rgb': target_rgb,
                        'solutions': [{
                            'solution_type': 'global_best',
                            'thickness': global_best.get('thickness', []),
                            'pred_rgb_amorphous': amorphous_rgb,
                            'pred_rgb_crystalline': crystalline_rgb,
                            'deltaE': global_best.get('deltaE', 0),
                            'deltaED': global_best.get('deltaED', 0)
                        }]
                    }
                    simplified_designs.append(color_design)

            if simplified_designs:
                self.selected_designs = simplified_designs
                self.static_results_window = StaticResultsWindow(simplified_designs, parent=self)
                self.static_results_window.show()
                self.hide()
            else:
                self.show_error("No static design results")
                self.reset_generate_btn()

        except Exception as e:
            print(f"Process static results failed: {e}")
            self.show_error(f"Process failed: {str(e)}")
            self.reset_generate_btn()

    def dynamic_results_confirmed(self, selected_designs):
        try:
            import sys
            print("[DEBUG] Step3Window.dynamic_results_confirmed called")
            safe_flush_stdout()
            self.selected_designs = selected_designs
            self.open_step4()
        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            with open('debug.log', 'a', encoding='utf-8') as f:
                f.write(f"dynamic_results_confirmed error:\n{error_msg}\n")
            QMessageBox.critical(self, "Error", f"Confirmation failed:\n{str(e)}")

    def static_results_confirmed(self, selected_designs):
        try:
            self.selected_designs = selected_designs
            QTimer.singleShot(10, self.open_step4)
        except Exception as e:
            print(f"Static results confirmation failed: {e}")
            self.show_error(f"Confirmation failed: {str(e)}")

    def open_step4(self):
        try:
            import sys
            print("[DEBUG] Step3Window.open_step4 called")
            safe_flush_stdout()
            self.hide()
            print("[DEBUG] Step3Window hidden")
            safe_flush_stdout()
            self.step4_window = Step4Window(self)
            print("[DEBUG] Step4Window instance created")
            safe_flush_stdout()
            self.step4_window.show()
            print("[DEBUG] Step4Window.show() executed")
            safe_flush_stdout()
        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            with open('debug.log', 'a', encoding='utf-8') as f:
                f.write(f"open_step4 error:\n{error_msg}\n")
            QMessageBox.critical(self, "Error", f"Cannot proceed to next step: {str(e)}")
            self.show_error(f"Cannot proceed: {str(e)}")

    def show_error(self, message):
        try:
            QMessageBox.critical(self, "Error", message)
        except:
            print(f"Error: {message}")

    def create_solution_dict(self, color_index, solution_type, solution_data, target_rgb, cluster_id):
        if not isinstance(solution_data, dict):
            solution_data = {}
        return {
            'color_index': color_index,
            'solution_type': solution_type,
            'cluster_id': cluster_id,
            'thickness': solution_data.get('thickness', []),
            'deltaE': solution_data.get('deltaE', 0),
            'deltaED': solution_data.get('deltaED', 0),
            'pred_lab_amorphous': solution_data.get('pred_lab_amorphous', [50, 0, 0]),
            'pred_lab_crystalline': solution_data.get('pred_lab_crystalline', [50, 0, 0]),
            'target_rgb': target_rgb,
            'cluster_size': solution_data.get('cluster_size', 1),
            'cluster_center_color': solution_data.get('cluster_center_color', []),
            'cluster_amorphous_center': solution_data.get('cluster_amorphous_center', [])
        }

    def convert_to_simplified_format(self, color_index, target_rgb, solutions):
        simplified_solutions = []
        for sol in solutions:
            if isinstance(sol, dict):
                amorphous_lab = sol.get('pred_lab_amorphous', [50, 0, 0])
                crystalline_lab = sol.get('pred_lab_crystalline', [50, 0, 0])
                amorphous_rgb = ColorConverter.lab_to_rgb(amorphous_lab)
                crystalline_rgb = ColorConverter.lab_to_rgb(crystalline_lab)
                simplified_solutions.append({
                    'solution_type': sol.get('solution_type', 'unknown'),
                    'thickness': sol.get('thickness', []),
                    'pred_rgb_amorphous': amorphous_rgb,
                    'pred_rgb_crystalline': crystalline_rgb,
                    'deltaE': sol.get('deltaE', 0),
                    'deltaED': sol.get('deltaED', 0)
                })
        return {
            'color_index': color_index,
            'target_rgb': target_rgb,
            'solutions': simplified_solutions
        }