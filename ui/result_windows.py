"""
Result-related windows - final simplified version; only essential widgets
kept, 1280x800 window, enlarged fonts.
"""
import sys
import time
import traceback

import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QRadioButton,
                             QButtonGroup, QFileDialog, QLineEdit,
                             QComboBox, QFrame, QScrollArea,
                             QMessageBox, QGroupBox, QGridLayout, QTabWidget, QTextEdit,
                             QTableWidget, QTableWidgetItem, QHeaderView, QDoubleSpinBox, QSplitter)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QColor, QBrush, QFont
from .widgets import ResultWindow, DynamicResultsWindow, StaticResultsWindow, ColorConverter
from .window_utils import apply_main_geometry, apply_popup_geometry
from .worker import run_with_progress
from utils.helpers import safe_flush_stdout, get_default_dir, get_image_dir


class ImageClusterWindow(QWidget):
    def __init__(self, colors, title="Color Clustering Results", parent=None):
        super().__init__()
        self.parent_window = parent
        self.colors = colors
        self.setWindowTitle(title)

        # Window size: height reduced by 1/4, width slightly increased to fit color blocks
        self.setGeometry(200, 200, 1200, 640)

        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(40, 40, 40, 40)

        # Title
        title_label = QLabel(title)
        title_label.setStyleSheet("font-size: 30px; font-weight: bold;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # Color count (enlarged font)
        count_label = QLabel(f"Total Colors: {len(colors)}")
        count_label.setStyleSheet("color: #666; font-size: 26px; font-weight: bold;")
        count_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(count_label)

        # Color block container - grid layout distributes blocks evenly
        scroll_area = QScrollArea()
        scroll_widget = QWidget()

        # Compute colors per row (based on window width)
        colors_per_row = min(4, len(colors))  # At most 4 per row
        color_layout = QGridLayout()
        color_layout.setSpacing(50)  # Increased spacing between color blocks
        color_layout.setContentsMargins(60, 30, 60, 30)  # Left/right margins keep content centered

        for i, color in enumerate(colors):
            row = i // colors_per_row
            col = i % colors_per_row
            color_widget = self.create_color_block(color, i)
            color_layout.addWidget(color_widget, row, col, Qt.AlignCenter)

        # Add stretch rows/columns to center the color blocks
        if len(colors) < colors_per_row:
            for col in range(len(colors), colors_per_row):
                color_layout.setColumnStretch(col, 1)
        color_layout.setRowStretch((len(colors) - 1) // colors_per_row + 1, 1)

        scroll_widget.setLayout(color_layout)
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumHeight(280)
        layout.addWidget(scroll_area)

        # Confirm button
        confirm_btn = QPushButton("Confirm")
        confirm_btn.setStyleSheet("""
            QPushButton {
                font-size: 26px;
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
        confirm_btn.clicked.connect(self.close)
        layout.addWidget(confirm_btn)

        self.setLayout(layout)

        # Confirmation popup: dedicated size, centered
        apply_popup_geometry(self)

    def create_color_block(self, color, index):
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(8)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setAlignment(Qt.AlignCenter)  # Center-align content

        # Color index label
        index_label = QLabel(f"Color {index + 1}")
        index_label.setAlignment(Qt.AlignCenter)
        index_label.setStyleSheet("font-weight: bold; font-size: 22px; color: #2c3e50;")
        layout.addWidget(index_label)

        # Color block
        color_label = QLabel()
        color_label.setFixedSize(160, 160)
        color_label.setStyleSheet("border: 2px solid #333; border-radius: 8px; background: transparent;")
        pixmap = QPixmap(160, 160)
        pixmap.fill(QColor(*color))
        color_label.setPixmap(pixmap)
        layout.addWidget(color_label)

        # RGB value label (no wrapping, width fits the card)
        rgb_text = f"RGB({color[0]}, {color[1]}, {color[2]})"
        rgb_label = QLabel(rgb_text)
        rgb_label.setAlignment(Qt.AlignCenter)
        rgb_label.setStyleSheet("font-size: 20px; color: #555; font-family: Consolas, monospace;")
        rgb_label.setFixedWidth(190)  # Fits the 200px card width
        rgb_label.setWordWrap(False)  # Disable word wrap
        rgb_label.setMinimumHeight(25)  # Ensure a minimum height
        layout.addWidget(rgb_label)

        widget.setLayout(layout)
        widget.setFixedSize(200, 280)  # Keep the original card size
        return widget


class ColorInputWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        self.parent_window = parent
        self.color_space = "RGB"
        self.color_groups = [None]  # Matches the initial "Group 1" combo item one-to-one
        self.current_group = 0
        self.input_container = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Color Input")
        apply_popup_geometry(self)  # Popup with dedicated size, centered

        layout = QVBoxLayout()
        layout.setSpacing(25)
        layout.setContentsMargins(80, 60, 80, 60)

        # Color space selection
        space_layout = QHBoxLayout()
        space_label = QLabel("Color Space:")
        space_label.setStyleSheet("font-size: 18px;")
        self.space_combo = QComboBox()
        self.space_combo.addItems(["RGB", "Lab", "XYZ"])
        self.space_combo.currentTextChanged.connect(self.update_input_fields)
        self.space_combo.setStyleSheet("font-size: 18px; padding: 8px;")
        space_layout.addWidget(space_label)
        space_layout.addWidget(self.space_combo)
        space_layout.addStretch()
        layout.addLayout(space_layout)

        # Color group management
        group_layout = QHBoxLayout()
        group_label = QLabel("Groups:")
        group_label.setStyleSheet("font-size: 18px;")
        self.group_combo = QComboBox()
        self.group_combo.addItem("Group 1")
        self.group_combo.currentIndexChanged.connect(self.switch_color_group)
        self.group_combo.setStyleSheet("font-size: 18px; padding: 8px;")
        add_group_btn = QPushButton("Add Group")
        add_group_btn.setStyleSheet("font-size: 16px; padding: 8px;")
        add_group_btn.clicked.connect(self.add_color_group)
        remove_group_btn = QPushButton("Remove Group")
        remove_group_btn.setStyleSheet("font-size: 16px; padding: 8px;")
        remove_group_btn.clicked.connect(self.remove_color_group)

        group_layout.addWidget(group_label)
        group_layout.addWidget(self.group_combo)
        group_layout.addWidget(add_group_btn)
        group_layout.addWidget(remove_group_btn)
        group_layout.addStretch()
        layout.addLayout(group_layout)

        # Input field container
        self.input_container = QWidget()
        self.input_layout = QVBoxLayout()
        self.input_container.setLayout(self.input_layout)
        layout.addWidget(self.input_container)

        # Entered group count info
        self.groups_info_label = QLabel("Entered: 0 / 0 groups")
        self.groups_info_label.setStyleSheet("font-size: 16px; color: #555;")
        layout.addWidget(self.groups_info_label)

        # Confirm button
        confirm_btn = QPushButton("Confirm All Colors")
        confirm_btn.setStyleSheet("""
            QPushButton {
                font-size: 22px;
                padding: 15px;
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        confirm_btn.clicked.connect(self.confirm_colors)
        layout.addWidget(confirm_btn)

        layout.addStretch()
        self.update_input_fields()
        self.setLayout(layout)

    def update_input_fields(self):
        while self.input_layout.count():
            child = self.input_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
            elif child.layout():
                self.clear_layout(child.layout())
                child.layout().deleteLater()

        space = self.space_combo.currentText()
        if space == "RGB":
            fields = ["R", "G", "B"]
            ranges = ["0-255", "0-255", "0-255"]
        elif space == "Lab":
            fields = ["L", "a", "b"]
            ranges = ["0-100", "-128-127", "-128-127"]
        else:  # XYZ
            fields = ["X", "Y", "Z"]
            ranges = ["0-1", "0-1", "0-1"]

        for field, range_text in zip(fields, ranges):
            hbox = QHBoxLayout()
            label = QLabel(f"{field}:")
            label.setStyleSheet("font-size: 18px;")
            line_edit = QLineEdit()
            line_edit.setPlaceholderText(f"({range_text})")
            line_edit.setStyleSheet("font-size: 18px; padding: 8px;")
            hbox.addWidget(label)
            hbox.addWidget(line_edit)
            hbox.addStretch()
            self.input_layout.addLayout(hbox)

    def clear_layout(self, layout):
        if layout is not None:
            while layout.count():
                child = layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
                elif child.layout():
                    self.clear_layout(child.layout())

    def switch_color_group(self, index):
        if 0 <= index < len(self.color_groups):
            self.current_group = index
            self.load_color_to_inputs(self.color_groups[index])

    def add_color_group(self):
        current_color = self.get_current_input_values()
        if current_color:
            # Save the current group's input (guard against index overflow)
            while len(self.color_groups) <= self.current_group:
                self.color_groups.append(None)
            self.color_groups[self.current_group] = current_color

        self.color_groups.append(None)
        # Numbering always follows the actual combo item count to avoid duplicates
        self.group_combo.addItem(f"Group {self.group_combo.count() + 1}")
        self.group_combo.setCurrentIndex(self.group_combo.count() - 1)
        self.clear_input_fields()
        self.update_groups_info()

    def remove_color_group(self):
        if len(self.color_groups) <= 1:
            return
        del self.color_groups[self.current_group]
        self.group_combo.removeItem(self.current_group)
        # Renumber after deletion so Group 1..N stays consecutive and unique
        for i in range(self.group_combo.count()):
            self.group_combo.setItemText(i, f"Group {i + 1}")
        if self.current_group >= len(self.color_groups):
            self.current_group = len(self.color_groups) - 1
        self.group_combo.setCurrentIndex(self.current_group)
        if self.color_groups:
            self.load_color_to_inputs(self.color_groups[self.current_group])
        self.update_groups_info()

    def get_current_input_values(self):
        space = self.space_combo.currentText()
        values = []
        for i in range(self.input_layout.count()):
            layout_item = self.input_layout.itemAt(i)
            if layout_item and layout_item.layout():
                inner_layout = layout_item.layout()
                if inner_layout.count() >= 2:
                    line_edit = inner_layout.itemAt(1).widget()
                    if line_edit and isinstance(line_edit, QLineEdit):
                        text = line_edit.text().strip()
                        if text:
                            try:
                                val = float(text)
                                values.append(val)
                            except ValueError:
                                return None
                        else:
                            return None
        if len(values) == 3:
            return {'space': space, 'values': values}
        return None

    def load_color_to_inputs(self, color_data):
        if not color_data:
            self.clear_input_fields()
            return
        for i in range(self.input_layout.count()):
            layout_item = self.input_layout.itemAt(i)
            if layout_item and layout_item.layout():
                inner_layout = layout_item.layout()
                if inner_layout.count() >= 2:
                    line_edit = inner_layout.itemAt(1).widget()
                    if line_edit and isinstance(line_edit, QLineEdit) and i < len(color_data['values']):
                        line_edit.setText(str(color_data['values'][i]))

    def clear_input_fields(self):
        for i in range(self.input_layout.count()):
            layout_item = self.input_layout.itemAt(i)
            if layout_item and layout_item.layout():
                inner_layout = layout_item.layout()
                if inner_layout.count() >= 2:
                    line_edit = inner_layout.itemAt(1).widget()
                    if line_edit and isinstance(line_edit, QLineEdit):
                        line_edit.clear()

    def update_groups_info(self):
        valid_groups = sum(1 for group in self.color_groups if group is not None)
        self.groups_info_label.setText(f"Entered: {valid_groups} / {len(self.color_groups)} groups")

    def confirm_colors(self):
        from core.color_processing import validate_multiple_colors

        current_color = self.get_current_input_values()
        if current_color:
            if len(self.color_groups) > self.current_group:
                self.color_groups[self.current_group] = current_color

        valid_colors = [group for group in self.color_groups if group is not None]

        if valid_colors:
            all_valid, error_messages = validate_multiple_colors(valid_colors)
            if all_valid:
                self.color_groups = valid_colors
                if self.parent_window:
                    self.parent_window.color_input_confirmed(self.color_groups)
                self.close()
            else:
                error_text = "\n".join(error_messages)
                QMessageBox.warning(self, "Input Error", f"Color validation failed:\n{error_text}")
        else:
            QMessageBox.warning(self, "Input Error", "At least one valid color group is required")





class Step4Window(QWidget):
    def __init__(self, prev_window):
        try:
            print("[DEBUG] Step4Window.__init__ started")
            safe_flush_stdout()

            super().__init__()
            self.prev_window = prev_window
            self.design_type = prev_window.design_type
            self.selected_designs = prev_window.selected_designs
            self.all_design_results = prev_window.all_design_results
            self.color_groups = prev_window.color_groups

            self.result_window = None
            self.environment_path = None
            self.generate_camouflage = True

            print("[DEBUG] Step4Window.__init__ calling init_ui")
            safe_flush_stdout()
            self.init_ui()
            self.update_status()
            self.init_budget_table()
            print("[DEBUG] Step4Window.__init__ completed")
            safe_flush_stdout()

        except Exception as e:
            error_msg = traceback.format_exc()
            with open('debug.log', 'a', encoding='utf-8') as f:
                f.write(f"Step4Window __init__ error:\n{error_msg}\n")
            QMessageBox.critical(None, "Error", f"Failed to initialize Step4 window:\n{str(e)}")
            raise

    def init_ui(self):
        try:
            print("[DEBUG] Step4Window.init_ui started")
            safe_flush_stdout()

            self.setWindowTitle("Step 4: Final Confirmation")
            apply_main_geometry(self)  # Apply unified main window size and center

            self.setStyleSheet("""
                QLabel { font-size: 18px; }
                QPushButton { font-size: 18px; padding: 8px 16px; }
                QGroupBox { 
                    font-size: 20px; 
                    font-weight: bold; 
                    margin-top: 12px; 
                    border: 1px solid #dee2e6;
                    border-radius: 6px;
                    padding-top: 10px;
                }
                QGroupBox::title { 
                    subcontrol-origin: margin; 
                    left: 10px; 
                    padding: 0 8px; 
                }
                QTabWidget::tab { 
                    font-size: 18px; 
                    padding: 8px 16px; 
                }
                QTabWidget::pane { 
                    border: 1px solid #dee2e6; 
                    border-radius: 6px; 
                    background-color: white; 
                }
                QTableWidget { 
                    font-size: 16px; 
                    gridline-color: #dee2e6;
                }
                QTableWidget::item { 
                    padding: 4px; 
                }
                QHeaderView::section { 
                    font-size: 16px; 
                    font-weight: bold; 
                    background-color: #f8f9fa; 
                    padding: 6px; 
                }
                QDoubleSpinBox { 
                    font-size: 16px; 
                    padding: 4px; 
                }
                QScrollArea { 
                    border: none; 
                }
            """)

            main_layout = QVBoxLayout()
            main_layout.setSpacing(15)
            main_layout.setContentsMargins(20, 20, 20, 20)

            self.status_label = QLabel("Loading design info...")
            self.status_label.setStyleSheet("""
                font-size: 20px; 
                font-weight: bold; 
                color: #2c3e50; 
                padding: 10px; 
                background-color: #ecf0f1; 
                border-radius: 6px;
            """)
            main_layout.addWidget(self.status_label)

            self.tab_widget = QTabWidget()
            self.tab_widget.setStyleSheet("""
                QTabBar::tab:selected { 
                    background-color: white; 
                    border-bottom: 3px solid #3498db; 
                }
            """)

            self.design_tab = QWidget()
            self.init_design_tab()
            self.tab_widget.addTab(self.design_tab, "📊 Design Results")

            self.camouflage_tab = QWidget()
            self.init_camouflage_tab()
            self.tab_widget.addTab(self.camouflage_tab, "🎨 Camouflage Budget")

            main_layout.addWidget(self.tab_widget)

            self.result_btn = QPushButton("Generate Final Results")
            self.result_btn.setStyleSheet("""
                QPushButton {
                    font-size: 22px;
                    padding: 12px 24px;
                    background-color: #3498db;
                    color: white;
                    border: none;
                    border-radius: 8px;
                }
                QPushButton:hover {
                    background-color: #2980b9;
                }
            """)
            self.result_btn.clicked.connect(self.generate_result)
            main_layout.addWidget(self.result_btn)

            self.setLayout(main_layout)

            QTimer.singleShot(50, self.update_status)

            print("[DEBUG] Step4Window.init_ui completed")
            safe_flush_stdout()

        except Exception as e:
            error_msg = traceback.format_exc()
            with open('debug.log', 'a', encoding='utf-8') as f:
                f.write(f"Step4Window init_ui error:\n{error_msg}\n")
            QMessageBox.critical(self, "Error", f"UI initialization failed:\n{str(e)}")
            raise

    def init_design_tab(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(8)

        self.details_scroll = QScrollArea()
        self.details_scroll.setWidgetResizable(True)
        self.details_scroll.setStyleSheet("border: none;")
        self.details_widget = QWidget()
        self.details_widget_layout = QVBoxLayout()
        self.details_widget_layout.setSpacing(8)
        self.details_widget.setLayout(self.details_widget_layout)
        self.details_scroll.setWidget(self.details_widget)

        layout.addWidget(self.details_scroll)
        self.design_tab.setLayout(layout)

    def init_camouflage_tab(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)

        info_label = QLabel("Set budget proportions (normalized automatically):")
        info_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #2c3e50;")
        layout.addWidget(info_label)

        self.budget_table = QTableWidget()
        self.budget_table.setColumnCount(3)
        self.budget_table.setHorizontalHeaderLabels(["Color", "Original Weight", "User Budget"])
        self.budget_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)
        self.budget_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.budget_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.budget_table.setColumnWidth(0, 180)  # Widen to fit larger color blocks
        self.budget_table.verticalHeader().setVisible(False)
        self.budget_table.verticalHeader().setDefaultSectionSize(60)  # Further increase row height
        layout.addWidget(self.budget_table)

        btn_layout = QHBoxLayout()
        reset_btn = QPushButton("Reset")
        reset_btn.clicked.connect(self.reset_budget_to_original)
        reset_btn.setStyleSheet("font-size: 16px; padding: 6px 12px;")
        normalize_btn = QPushButton("Normalize")
        normalize_btn.clicked.connect(self.normalize_budget)
        normalize_btn.setStyleSheet("font-size: 16px; padding: 6px 12px;")
        btn_layout.addWidget(reset_btn)
        btn_layout.addWidget(normalize_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        self.sum_label = QLabel("Current Sum: 1.0000")
        self.sum_label.setStyleSheet("font-size: 16px; color: #333;")
        layout.addWidget(self.sum_label)

        layout.addStretch()
        self.camouflage_tab.setLayout(layout)

    def init_budget_table(self):
        color_count = len(self.color_groups)
        self.budget_table.setRowCount(color_count)

        self.budget_spinboxes = []

        for i, group in enumerate(self.color_groups):
            # Color column: index + color block (further enlarged block)
            color_widget = QWidget()
            color_layout = QHBoxLayout()
            color_layout.setContentsMargins(4, 2, 4, 2)
            color_layout.setSpacing(12)

            index_label = QLabel(f"{i+1}")
            index_label.setStyleSheet("font-size: 16px; font-weight: bold;")
            color_layout.addWidget(index_label)

            color_label = QLabel()
            color_label.setFixedSize(60, 36)  # Further enlarged
            rgb = group['values']
            color_label.setStyleSheet(f"background-color: rgb({rgb[0]},{rgb[1]},{rgb[2]}); border: 1px solid gray; border-radius: 3px;")
            color_layout.addWidget(color_label)

            color_layout.addStretch()
            color_widget.setLayout(color_layout)
            self.budget_table.setCellWidget(i, 0, color_widget)

            # Original weight (font further reduced)
            weight = group.get('weight')
            if weight is None:
                weight = 1.0 / color_count if color_count > 0 else 1.0
            weight_item = QTableWidgetItem(f"{weight:.4f}")
            weight_item.setTextAlignment(Qt.AlignCenter)
            weight_item.setFlags(weight_item.flags() & ~Qt.ItemIsEditable)
            weight_item.setFont(QFont("", 12))  # Reduce font to 12px
            self.budget_table.setItem(i, 1, weight_item)

            # User budget input
            spinbox = QDoubleSpinBox()
            spinbox.setRange(0.0, 1.0)
            spinbox.setSingleStep(0.01)
            spinbox.setDecimals(4)
            spinbox.setValue(weight)
            spinbox.setAlignment(Qt.AlignCenter)
            spinbox.valueChanged.connect(self.update_budget_sum)
            self.budget_table.setCellWidget(i, 2, spinbox)
            self.budget_spinboxes.append(spinbox)

        self.update_budget_sum()

    def update_budget_sum(self):
        total = sum(spinbox.value() for spinbox in self.budget_spinboxes)
        self.sum_label.setText(f"Current Sum: {total:.4f}")

    def reset_budget_to_original(self):
        for i, spinbox in enumerate(self.budget_spinboxes):
            weight_item = self.budget_table.item(i, 1)
            if weight_item:
                original = float(weight_item.text())
                spinbox.setValue(original)
        self.update_budget_sum()

    def normalize_budget(self):
        values = [spinbox.value() for spinbox in self.budget_spinboxes]
        total = sum(values)
        if total == 0:
            QMessageBox.warning(self, "Warning", "Budget sum cannot be zero")
            return
        normalized = [v / total for v in values]
        for spinbox, norm in zip(self.budget_spinboxes, normalized):
            spinbox.setValue(norm)
        self.update_budget_sum()

    def get_current_budget(self):
        return [spinbox.value() for spinbox in self.budget_spinboxes]

    def update_status(self):
        try:
            design_type_text = "Dynamic" if self.design_type == 'dynamic' else "Static"
            if self.design_type == 'dynamic':
                selected_count = len(self.selected_designs)
                colors_count = len(set(design.get('color_index', 0) for design in self.selected_designs))
                status_text = f"Type: {design_type_text} | Colors: {colors_count} | Solutions: {selected_count}"
            else:
                colors_count = len(self.selected_designs)
                status_text = f"Type: {design_type_text} | Colors: {colors_count} | Optimal solutions"

            total_solutions = sum(len(color.get('solutions', [])) for color in self.selected_designs)
            if total_solutions > 0:
                total_deltaE = sum(
                    sum(sol.get('deltaE', 0) for sol in color.get('solutions', []))
                    for color in self.selected_designs
                )
                avg_deltaE = total_deltaE / total_solutions
                status_text += f" | Avg ΔE: {avg_deltaE:.2f}"

            self.status_label.setText(status_text)
            self.update_design_details()
        except Exception as e:
            print(f"Status update failed: {e}")

    def update_design_details(self):
        try:
            while self.details_widget_layout.count():
                child = self.details_widget_layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
                elif child.layout():
                    self.clear_layout(child.layout())
                    child.layout().deleteLater()

            for color_design in self.selected_designs:
                color_idx = color_design.get('color_index', 0)
                solutions = color_design.get('solutions', [])
                if solutions:
                    target_rgb = color_design.get('target_rgb', [0, 0, 0])
                    group_title = f"Color {color_idx+1}  |  Target RGB: ({target_rgb[0]}, {target_rgb[1]}, {target_rgb[2]})"
                    color_group = QGroupBox(group_title)
                    color_group.setStyleSheet("font-size: 20px; font-weight: bold;")  # Enlarged group title
                    color_layout = QVBoxLayout()
                    color_layout.setSpacing(8)

                    for i, solution in enumerate(solutions):
                        design_widget = self.create_design_widget(solution, i, color_idx)
                        color_layout.addWidget(design_widget)

                    color_group.setLayout(color_layout)
                    self.details_widget_layout.addWidget(color_group)

            self.details_widget_layout.addStretch()
        except Exception as e:
            print(f"Update design details failed: {e}")

    def create_design_widget(self, solution, index, color_index):
        widget = QWidget()
        widget.setStyleSheet("""
            QWidget {
                background-color: #f9f9f9;
                border: 1px solid #e0e0e0;
                border-radius: 6px;
            }
        """)
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        sol_type = solution.get('solution_type', 'unknown')
        cluster_id = solution.get('cluster_id')
        # cluster_best shows the real cluster id; other types have no cluster suffix
        if sol_type == 'cluster_best' and cluster_id is not None and cluster_id >= 0:
            type_display = f"Cluster Best (Cluster {cluster_id + 1})"
        else:
            type_display = self.translate_solution_type(sol_type)

        type_label = QLabel(f"Solution {index+1}: {type_display}")
        type_label.setStyleSheet("font-weight: bold; color: #0066cc; font-size: 22px;")  # Title further enlarged
        layout.addWidget(type_label)

        grid = QGridLayout()
        grid.setHorizontalSpacing(20)
        grid.setVerticalSpacing(8)

        deltae = solution.get('deltaE', 0)
        deltae_label = QLabel(f"ΔE: {deltae:.3f}")
        deltae_label.setStyleSheet("font-size: 20px;")  # Content font enlarged to 20px
        grid.addWidget(deltae_label, 0, 0)

        deltaed = solution.get('deltaED', 0)
        deltaed_label = QLabel(f"ΔED: {deltaed:.3f}")
        deltaed_label.setStyleSheet("font-size: 20px;")
        grid.addWidget(deltaed_label, 0, 1)

        thickness = solution.get('thickness', [])
        if thickness:
            thickness_text = ColorConverter.format_thickness(thickness)
            thickness_label = QLabel(f"Thickness: {thickness_text}")
            thickness_label.setStyleSheet("font-size: 20px;")
            grid.addWidget(thickness_label, 1, 0, 1, 2)

        amorphous_rgb = solution.get('pred_rgb_amorphous', [128, 128, 128])
        amorphous_label = QLabel(f"Amorphous RGB: ({amorphous_rgb[0]}, {amorphous_rgb[1]}, {amorphous_rgb[2]})")
        amorphous_label.setStyleSheet("font-size: 20px;")
        grid.addWidget(amorphous_label, 2, 0, 1, 2)

        crystalline_rgb = solution.get('pred_rgb_crystalline', [128, 128, 128])
        crystalline_label = QLabel(f"Crystalline RGB: ({crystalline_rgb[0]}, {crystalline_rgb[1]}, {crystalline_rgb[2]})")
        crystalline_label.setStyleSheet("font-size: 20px;")
        grid.addWidget(crystalline_label, 3, 0, 1, 2)

        layout.addLayout(grid)
        widget.setLayout(layout)
        return widget

    def translate_solution_type(self, sol_type):
        mapping = {
            'multilayer': 'Cluster Solution',
            'cluster_best': 'Cluster Best',
            'global_best': 'Global Best',
            'max_deltaED': 'Max ΔED',
            'deltaE_max': 'ΔE Max Solution',
        }
        return mapping.get(sol_type, sol_type.replace('_', ' ').title())

    def clear_layout(self, layout):
        if layout is not None:
            while layout.count():
                child = layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
                elif child.layout():
                    self.clear_layout(child.layout())

    def select_environment(self):
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Select Environment Image", get_image_dir(),
                "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff *.tif);;All Files (*.*)"
            )
            if file_path:
                import os
                if not os.path.exists(file_path):
                    QMessageBox.warning(self, "Warning", f"File does not exist: {file_path}")
                    return

                try:
                    with open(file_path, 'rb') as f:
                        header = f.read(10)
                        if not header:
                            raise ValueError("Empty file")
                except Exception as e:
                    QMessageBox.warning(self, "Warning", f"Cannot read file: {str(e)}")
                    return

                import imghdr
                file_type = imghdr.what(file_path)
                supported_types = ['png', 'jpeg', 'jpg', 'bmp', 'tiff']
                if file_type not in supported_types:
                    QMessageBox.warning(self, "Warning",
                                         f"Unsupported format: {file_type}\nSupported: {', '.join(supported_types)}")
                    return

                self.environment_path = file_path
                short_path = os.path.basename(file_path)

                try:
                    import cv2
                    test_img = cv2.imread(file_path)
                    if test_img is None:
                        QMessageBox.warning(self, "Warning",
                                            "OpenCV cannot read the image. Please check format or use another image.")
                        self.environment_path = None
                    else:
                        height, width, channels = test_img.shape
                        print(f"Environment selected: {short_path} ({width}×{height})")
                except Exception as e:
                    print(f"Image test failed: {e}")
        except Exception as e:
            print(f"Select environment failed: {e}")
            QMessageBox.warning(self, "Warning", f"Failed to select environment image: {str(e)}")

    def generate_result(self):
        """Generate final results (spectrum precomputation + camouflage generation) in a background thread; main thread shows progress"""
        try:
            budget = self.get_current_budget()
            total = sum(budget)
            if total == 0:
                QMessageBox.warning(self, "Warning", "Budget sum cannot be zero")
                return
            budget_normalized = [b / total for b in budget]
            design_params = {
                'design_type': self.design_type,
                'selected_designs': self.selected_designs,
                'budget': budget_normalized
            }

            self.result_btn.setEnabled(False)
            self.result_btn.setText("Generating...")
            self._result_start = time.perf_counter()

            selected_designs = self.selected_designs

            def work(report, worker):
                # Precompute infrared spectra so the main thread does not stall when the result window opens the spectrum tab
                spectrum_data = {}
                try:
                    report(-1, "Computing infrared spectra (TMM)...")
                    from core.spectrum_calculation import calculate_all_spectra
                    spectrum_data = calculate_all_spectra(
                        selected_designs,
                        wavelength_range=(3000, 14000, 10)  # 3-14um
                    )
                except Exception as e:
                    print(f"Spectrum pre-calculation failed: {e}")
                    spectrum_data = {}

                report(-1, "Generating camouflage pattern...")
                result_data = self.generate_design_result(design_params)
                if result_data is not None:
                    result_data['spectrum_results'] = spectrum_data
                return result_data

            def on_success(result_data):
                self.result_btn.setEnabled(True)
                self.result_btn.setText("Generate Final Results")

                if result_data:
                    # Total design time = Step3 design generation time + Step4 result generation time
                    result_elapsed = time.perf_counter() - self._result_start
                    step3_elapsed = getattr(self.prev_window, 'design_elapsed', 0)
                    result_data['design_time_seconds'] = round(step3_elapsed + result_elapsed, 2)
                    print(f"Total design time: {result_data['design_time_seconds']:.1f} s")

                    main_window = None
                    try:
                        curr = self.prev_window
                        for _ in range(5):
                            if hasattr(curr, 'main_window'):
                                main_window = curr.main_window
                                break
                            elif hasattr(curr, 'prev_window'):
                                curr = curr.prev_window
                            else:
                                break
                    except Exception:
                        main_window = None

                    self.result_window = ResultWindow(result_data, main_window)
                    self.result_window.show()
                    self.hide()
                else:
                    QMessageBox.warning(self, "Warning", "No design results generated")

            def on_error(err):
                print(f"Generate result failed: {err}")
                self.result_btn.setEnabled(True)
                self.result_btn.setText("Generate Final Results")
                QMessageBox.critical(self, "Error", f"Failed to generate results: {err}")

            run_with_progress(self, "Generating Final Results",
                              "Preparing...",
                              work, on_success, on_error)

        except Exception as e:
            print(f"Generate result failed: {e}")
            self.result_btn.setEnabled(True)
            self.result_btn.setText("Generate Final Results")
            QMessageBox.critical(self, "Error", f"Failed to start generation: {str(e)}")

    def generate_design_result(self, design_params):
        try:
            result_data = design_params.copy()
            designs_data = design_params.get('selected_designs', [])
            if designs_data:
                result_data['design_type'] = design_params.get('design_type', 'static')
                colors_count = len(designs_data)
                solutions_count = sum(len(color_design['solutions']) for color_design in designs_data)
                result_data['colors_count'] = colors_count
                result_data['solutions_count'] = solutions_count
                print(f"Design data: {designs_data[0] if designs_data else 'empty'}")

                result_data['design_details'] = []
                for color_design in designs_data:
                    color_index = color_design.get('color_index', 0)
                    target_rgb = color_design.get('target_rgb', [128, 128, 128])
                    for solution in color_design.get('solutions', []):
                        if isinstance(solution, dict):
                            design_detail = {
                                'color_index': color_index,
                                'solution_type': solution.get('solution_type', 'unknown'),
                                'thickness': solution.get('thickness', []),
                                'deltaE': solution.get('deltaE', 0),
                                'deltaED': solution.get('deltaED', 0),
                                'pred_rgb_amorphous': solution.get('pred_rgb_amorphous', [128, 128, 128]),
                                'pred_rgb_crystalline': solution.get('pred_rgb_crystalline', [128, 128, 128]),
                                'target_rgb': target_rgb
                            }
                            result_data['design_details'].append(design_detail)
                print(f"Generated design details count: {len(result_data['design_details'])}")

                camouflage_data = self.generate_camouflage_pattern_data(result_data)
                if camouflage_data:
                    result_data.update(camouflage_data)
                else:
                    print("Warning: Camouflage pattern generation failed or returned no data")

                return result_data
        except Exception as e:
            print(f"Generate design data failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def generate_camouflage_pattern_data(self, design_data):
        try:
            from Camo.main import generate_camouflage_pattern

            selected_designs = design_data.get('selected_designs', [])
            budget = design_data.get('budget', [])

            if not selected_designs:
                print("Warning: No designs selected for camouflage")
                return None

            amorphous_colors = []
            crystalline_colors = []
            used_budget = []
            used_indices = []

            for i, color_design in enumerate(selected_designs):
                solutions = color_design.get('solutions', [])
                if not solutions:
                    continue
                solution = solutions[0]
                amorphous_rgb = solution.get('pred_rgb_amorphous')
                crystalline_rgb = solution.get('pred_rgb_crystalline')

                if amorphous_rgb is None:
                    amorphous_lab = solution.get('pred_lab_amorphous')
                    if amorphous_lab:
                        amorphous_rgb = ColorConverter.lab_to_rgb(amorphous_lab)
                    else:
                        continue
                if crystalline_rgb is None:
                    crystalline_lab = solution.get('pred_lab_crystalline')
                    if crystalline_lab:
                        crystalline_rgb = ColorConverter.lab_to_rgb(crystalline_lab)
                    else:
                        continue

                amorphous_colors.append(amorphous_rgb)
                crystalline_colors.append(crystalline_rgb)
                if i < len(budget):
                    used_budget.append(budget[i])
                else:
                    used_budget.append(1.0 / len(selected_designs))
                used_indices.append(i)

            if not amorphous_colors:
                print("Warning: No valid color data for camouflage generation")
                return None

            total_budget = sum(used_budget)
            if total_budget > 0:
                used_budget = [b / total_budget for b in used_budget]
            else:
                used_budget = [1.0 / len(used_budget)] * len(used_budget)

            camouflage_results = generate_camouflage_pattern(
                amorphous_colors=amorphous_colors,
                crystalline_colors=crystalline_colors,
                color_budget=used_budget
            )

            if camouflage_results:
                camouflage_results['color_solutions'] = [
                    {
                        'color_index': selected_designs[idx].get('color_index', idx),
                        'solution_type': selected_designs[idx]['solutions'][0].get('solution_type', 'unknown'),
                        'deltaE': selected_designs[idx]['solutions'][0].get('deltaE', 0),
                        'deltaED': selected_designs[idx]['solutions'][0].get('deltaED', 0),
                        'thickness': selected_designs[idx]['solutions'][0].get('thickness', [])
                    }
                    for idx in used_indices
                ]
                camouflage_results['environment_used'] = False
                camouflage_results['color_budget'] = used_budget
                return camouflage_results
            else:
                return None

        except ImportError as e:
            # Note: this function runs in a background thread; it must not create Qt widgets, so errors are printed to the console
            print(f"Import camouflage module failed: {e}")
            return None
        except Exception as e:
            print(f"Generate camouflage pattern failed: {e}")
            import traceback
            traceback.print_exc()
            return None