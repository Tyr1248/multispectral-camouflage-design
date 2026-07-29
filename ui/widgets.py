"""
自定义控件模块 - 最终简化版，仅保留必要控件，窗口1280x800，字体放大
包含动态结果窗口、静态结果窗口和最终结果窗口
"""
import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QCheckBox,
                             QGroupBox, QScrollArea, QFrame,
                             QTabWidget, QApplication, QMessageBox,
                             QTableWidget, QTableWidgetItem, QHeaderView,
                             QRadioButton, QButtonGroup, QComboBox, QTextEdit,
                             QGridLayout, QSizePolicy)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QColor, QImage, QFont, QBrush, QPainter, QPen
import matplotlib
from matplotlib import pyplot as plt
import re

matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# 尝试导入colour-science库，如果失败则使用备用方案
try:
    import colour
    from colour.models import Lab_to_XYZ, XYZ_to_sRGB, sRGB_to_XYZ, XYZ_to_Lab
    from colour.utilities import as_float_array, domain_range_scale
    COLOUR_AVAILABLE = True
    print("colour-science library available")
except ImportError:
    COLOUR_AVAILABLE = False
    print("Warning: colour-science library not available, using simplified color conversion")


class ColorConverter:
    """颜色空间转换器，使用colour-science库或备用方案"""

    @staticmethod
    def rgb_to_hex(rgb):
        """RGB转十六进制颜色"""
        if not rgb or len(rgb) < 3:
            return "#000000"
        return "#{:02x}{:02x}{:02x}".format(
            int(rgb[0]), int(rgb[1]), int(rgb[2])
        )

    @staticmethod
    def format_color_info(color_data):
        """格式化颜色信息显示"""
        if isinstance(color_data, dict):
            if 'rgb' in color_data:
                rgb = color_data['rgb']
                return f"RGB({rgb[0]}, {rgb[1]}, {rgb[2]}) {ColorConverter.rgb_to_hex(rgb)}"
            elif 'lab' in color_data:
                return f"Lab({color_data['lab'][0]:.1f}, {color_data['lab'][1]:.1f}, {color_data['lab'][2]:.1f})"
        elif isinstance(color_data, (list, tuple, np.ndarray)):
            if len(color_data) >= 3:
                if all(isinstance(c, (int, np.integer)) for c in color_data[:3]):
                    return f"RGB({color_data[0]}, {color_data[1]}, {color_data[2]}) {ColorConverter.rgb_to_hex(color_data[:3])}"
                else:
                    return f"Lab({color_data[0]:.1f}, {color_data[1]:.1f}, {color_data[2]:.1f})"
        return "Unknown color"

    @staticmethod
    def rgb_to_lab(rgb):
        """RGB转Lab颜色空间（标准范围）"""
        if not COLOUR_AVAILABLE:
            return ColorConverter._simple_rgb_to_lab(rgb)
        try:
            rgb_values = ColorConverter._extract_rgb_values(rgb)
            if rgb_values is None:
                return [50.0, 0.0, 0.0]
            rgb_array = np.array(rgb_values, dtype=np.float32)
            if rgb_array.max() > 1.0:
                rgb_array = rgb_array / 255.0
            xyz = sRGB_to_XYZ(rgb_array)
            lab = XYZ_to_Lab(xyz)
            return [float(lab[0]), float(lab[1]), float(lab[2])]
        except Exception as e:
            print(f"RGB to Lab conversion failed: {e}")
            return ColorConverter._simple_rgb_to_lab(rgb_values)

    @staticmethod
    def lab_to_rgb(lab):
        """Lab转RGB颜色空间（标准范围）"""
        if not COLOUR_AVAILABLE:
            return ColorConverter._simple_lab_to_rgb(lab)
        try:
            lab_values = ColorConverter._extract_lab_values(lab)
            if lab_values is None:
                return [128, 128, 128]
            lab_array = np.array(lab_values, dtype=np.float32)
            xyz = Lab_to_XYZ(lab_array)
            rgb = XYZ_to_sRGB(xyz)
            rgb = np.clip(rgb, 0.0, 1.0)
            rgb_255 = np.round(rgb * 255).astype(np.int32)
            return [int(rgb_255[0]), int(rgb_255[1]), int(rgb_255[2])]
        except Exception as e:
            print(f"Lab to RGB conversion failed: {e}")
            return ColorConverter._simple_lab_to_rgb(lab_values)

    @staticmethod
    def _extract_rgb_values(rgb_data):
        """从不同格式的输入中提取RGB值"""
        if rgb_data is None:
            return None
        if isinstance(rgb_data, (list, tuple, np.ndarray)):
            if len(rgb_data) >= 3:
                try:
                    return [float(rgb_data[0]), float(rgb_data[1]), float(rgb_data[2])]
                except (ValueError, TypeError):
                    return None
        elif isinstance(rgb_data, dict):
            for key in ['values', 'rgb', 'pred_rgb_amorphous', 'pred_rgb_crystalline', 'target_rgb']:
                if key in rgb_data and isinstance(rgb_data[key], (list, tuple, np.ndarray)):
                    if len(rgb_data[key]) >= 3:
                        try:
                            return [float(rgb_data[key][0]), float(rgb_data[key][1]), float(rgb_data[key][2])]
                        except (ValueError, TypeError):
                            continue
        elif isinstance(rgb_data, str):
            if rgb_data.startswith('RGB'):
                import re
                numbers = re.findall(r'\d+', rgb_data)
                if len(numbers) >= 3:
                    try:
                        return [float(numbers[0]), float(numbers[1]), float(numbers[2])]
                    except (ValueError, TypeError):
                        return None
        return None

    @staticmethod
    def _extract_lab_values(lab_data):
        """从不同格式的输入中提取Lab值"""
        if lab_data is None:
            return None
        if isinstance(lab_data, (list, tuple, np.ndarray)):
            if len(lab_data) >= 3:
                try:
                    return [float(lab_data[0]), float(lab_data[1]), float(lab_data[2])]
                except (ValueError, TypeError):
                    return None
        elif isinstance(lab_data, dict):
            for key in ['values', 'lab', 'pred_lab_amorphous', 'pred_lab_crystalline']:
                if key in lab_data and isinstance(lab_data[key], (list, tuple, np.ndarray)):
                    if len(lab_data[key]) >= 3:
                        try:
                            return [float(lab_data[key][0]), float(lab_data[key][1]), float(lab_data[key][2])]
                        except (ValueError, TypeError):
                            continue
        elif isinstance(lab_data, str):
            if lab_data.startswith('Lab'):
                import re
                numbers = re.findall(r'[-+]?\d*\.\d+|\d+', lab_data)
                if len(numbers) >= 3:
                    try:
                        return [float(numbers[0]), float(numbers[1]), float(numbers[2])]
                    except (ValueError, TypeError):
                        return None
        return None

    @staticmethod
    def _simple_rgb_to_lab(rgb):
        """简化的RGB到Lab转换（备用方案）"""
        try:
            if rgb is None:
                return [50.0, 0.0, 0.0]
            rgb_values = ColorConverter._extract_rgb_values(rgb)
            if not rgb_values:
                return [50.0, 0.0, 0.0]
            r, g, b = [max(0, min(255, c)) / 255.0 for c in rgb_values[:3]]
            r = r if r <= 0.04045 else ((r + 0.055) / 1.055) ** 2.4
            g = g if g <= 0.04045 else ((g + 0.055) / 1.055) ** 2.4
            b = b if b <= 0.04045 else ((b + 0.055) / 1.055) ** 2.4
            x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
            y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
            z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
            xn, yn, zn = 0.95047, 1.00000, 1.08883

            def f(t):
                if t > 0.008856:
                    return t ** (1 / 3)
                else:
                    return 7.787 * t + 16 / 116

            fx = f(x / xn)
            fy = f(y / yn)
            fz = f(z / zn)
            l = 116 * fy - 16
            a = 500 * (fx - fy)
            b_val = 200 * (fy - fz)
            return [max(0, min(100, l)), max(-128, min(127, a)), max(-128, min(127, b_val))]
        except Exception as e:
            print(f"Simplified RGB to Lab conversion failed: {e}")
            return [50.0, 0.0, 0.0]

    @staticmethod
    def _simple_lab_to_rgb(lab):
        """简化的Lab到RGB转换（备用方案）"""
        try:
            if lab is None:
                return [128, 128, 128]
            lab_values = ColorConverter._extract_lab_values(lab)
            if not lab_values:
                return [128, 128, 128]
            l, a, b = [float(c) for c in lab_values[:3]]
            l = max(0, min(100, l))
            a = max(-128, min(127, a))
            b_val = max(-128, min(127, b))

            def f_inv(t):
                if t > 0.2068966:
                    return t ** 3
                else:
                    return (t - 16 / 116) / 7.787

            xn, yn, zn = 0.95047, 1.00000, 1.08883
            fy = (l + 16) / 116
            fx = fy + (a / 500)
            fz = fy - (b_val / 200)
            x = xn * f_inv(fx)
            y = yn * f_inv(fy)
            z = zn * f_inv(fz)
            r_linear = x * 3.2404542 + y * -1.5371385 + z * -0.4985314
            g_linear = x * -0.9692660 + y * 1.8760108 + z * 0.0415560
            b_linear = x * 0.0556434 + y * -0.2040259 + z * 1.0572252

            def gamma_correct(t):
                if t <= 0.0031308:
                    return 12.92 * t
                else:
                    return 1.055 * (t ** (1 / 2.4)) - 0.055

            r = gamma_correct(r_linear)
            g = gamma_correct(g_linear)
            b = gamma_correct(b_linear)
            rgb = [max(0, min(1, c)) * 255 for c in [r, g, b]]
            return [int(round(c)) for c in rgb]
        except Exception as e:
            print(f"Simplified Lab to RGB conversion failed: {e}")
            return [128, 128, 128]

    @staticmethod
    def get_color_difference(lab1, lab2):
        """计算两个Lab颜色之间的ΔE*ab色差"""
        try:
            if not lab1 or not lab2 or len(lab1) < 3 or len(lab2) < 3:
                return 0
            return np.sqrt(
                (lab1[0] - lab2[0]) ** 2 +
                (lab1[1] - lab2[1]) ** 2 +
                (lab1[2] - lab2[2]) ** 2
            )
        except Exception as e:
            print(f"ΔE calculation failed: {e}")
            return 0

    @staticmethod
    def format_lab(lab):
        """格式化Lab值显示"""
        if not lab or len(lab) < 3:
            return "L: -, a: -, b: -"
        return f"L: {lab[0]:.1f}, a: {lab[1]:.1f}, b: {lab[2]:.1f}"

    @staticmethod
    def format_thickness(thickness_list):
        """格式化厚度列表显示"""
        if not thickness_list:
            return "N/A"
        if isinstance(thickness_list, (int, float)):
            return f"{thickness_list:.1f} nm"
        if hasattr(thickness_list, 'tolist'):
            thickness_list = thickness_list.tolist()
        thickness_strs = []
        for i, thickness in enumerate(thickness_list):
            if isinstance(thickness, (list, tuple)):
                if thickness:
                    thickness = thickness[0]
                else:
                    thickness = 0
            try:
                thickness_float = float(thickness)
                thickness_strs.append(f"{thickness_float:.1f}")
            except (ValueError, TypeError):
                thickness_strs.append("0.0")
        return f"[{', '.join(thickness_strs)}] nm"


class ColorDisplayWidget(QWidget):
    """颜色显示控件 - 简化版，仅显示颜色块和简要数值"""
    def __init__(self, label, color_data, is_target=False, parent=None):
        super().__init__(parent)
        self.label = label
        self.is_target = is_target
        self.color_rgb = None
        self.color_lab = None
        self.parse_color_data(color_data)
        self.init_ui()

    def parse_color_data(self, color_data):
        """解析颜色数据"""
        # 初始化默认值
        self.color_rgb = [128, 128, 128]
        self.color_lab = [50.0, 0.0, 0.0]

        try:
            if isinstance(color_data, dict):
                # 如果是字典，可能包含RGB和Lab
                if 'values' in color_data:
                    color_space = color_data.get('space', 'RGB')
                    if color_space == 'RGB':
                        rgb_values = ColorConverter._extract_rgb_values(color_data['values'])
                        if rgb_values:
                            self.color_rgb = [int(c) for c in rgb_values]
                            self.color_lab = ColorConverter.rgb_to_lab(self.color_rgb)
                    elif color_space == 'Lab':
                        lab_values = ColorConverter._extract_lab_values(color_data['values'])
                        if lab_values:
                            self.color_lab = [float(c) for c in lab_values]
                            self.color_rgb = ColorConverter.lab_to_rgb(self.color_lab)
                elif 'rgb' in color_data:
                    rgb_values = ColorConverter._extract_rgb_values(color_data['rgb'])
                    if rgb_values:
                        self.color_rgb = [int(c) for c in rgb_values]
                        self.color_lab = ColorConverter.rgb_to_lab(self.color_rgb)
                elif 'pred_rgb_amorphous' in color_data:
                    rgb_values = ColorConverter._extract_rgb_values(color_data['pred_rgb_amorphous'])
                    if rgb_values:
                        self.color_rgb = [int(c) for c in rgb_values]
                        self.color_lab = ColorConverter.rgb_to_lab(self.color_rgb)
                elif 'pred_rgb_crystalline' in color_data:
                    rgb_values = ColorConverter._extract_rgb_values(color_data['pred_rgb_crystalline'])
                    if rgb_values:
                        self.color_rgb = [int(c) for c in rgb_values]
                        self.color_lab = ColorConverter.rgb_to_lab(self.color_rgb)
                elif 'lab' in color_data:
                    lab_values = ColorConverter._extract_lab_values(color_data['lab'])
                    if lab_values:
                        self.color_lab = [float(c) for c in lab_values]
                        self.color_rgb = ColorConverter.lab_to_rgb(self.color_lab)
                elif 'pred_lab_amorphous' in color_data:
                    lab_values = ColorConverter._extract_lab_values(color_data['pred_lab_amorphous'])
                    if lab_values:
                        self.color_lab = [float(c) for c in lab_values]
                        self.color_rgb = ColorConverter.lab_to_rgb(self.color_lab)
                elif 'pred_lab_crystalline' in color_data:
                    lab_values = ColorConverter._extract_lab_values(color_data['pred_lab_crystalline'])
                    if lab_values:
                        self.color_lab = [float(c) for c in lab_values]
                        self.color_rgb = ColorConverter.lab_to_rgb(self.color_lab)
            elif isinstance(color_data, (list, tuple, np.ndarray)):
                if len(color_data) >= 3:
                    # 尝试判断是RGB还是Lab
                    first_val = color_data[0]
                    if isinstance(first_val, (int, np.integer)) or (
                            isinstance(first_val, (float, np.floating)) and first_val > 1.0):
                        # 可能是RGB值（整数或大于1的浮点数）
                        rgb_values = ColorConverter._extract_rgb_values(color_data)
                        if rgb_values:
                            self.color_rgb = [int(c) for c in rgb_values]
                            self.color_lab = ColorConverter.rgb_to_lab(self.color_rgb)
                    else:
                        # 可能是Lab值
                        lab_values = ColorConverter._extract_lab_values(color_data)
                        if lab_values:
                            self.color_lab = [float(c) for c in lab_values]
                            self.color_rgb = ColorConverter.lab_to_rgb(self.color_lab)

        except Exception as e:
            print(f"解析颜色数据失败: {e}, 数据: {color_data}")
            # 使用默认值
            self.color_rgb = [128, 128, 128]
            self.color_lab = [50.0, 0.0, 0.0]

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(5)

        label_widget = QLabel(self.label)
        if self.is_target:
            label_widget.setStyleSheet("font-weight: bold; font-size: 14px; color: #2c3e50; background-color: #e8f4fd; padding: 4px; border-radius: 3px;")
        else:
            label_widget.setStyleSheet("font-weight: bold; font-size: 14px; color: #2c3e50;")
        label_widget.setAlignment(Qt.AlignCenter)
        layout.addWidget(label_widget)

        self.color_block = QLabel()
        self.color_block.setFixedSize(90, 90)
        self.color_block.setStyleSheet("border: 2px solid #bdc3c7; border-radius: 8px;")
        self.update_color_display()
        layout.addWidget(self.color_block)

        info_widget = QWidget()
        info_layout = QVBoxLayout()
        info_layout.setSpacing(2)

        if self.color_rgb:
            rgb_text = f"RGB: {self.color_rgb[0]}, {self.color_rgb[1]}, {self.color_rgb[2]}"
            rgb_label = QLabel(rgb_text)
            rgb_label.setStyleSheet("font-size: 12px; color: #7f8c8d;")
            rgb_label.setAlignment(Qt.AlignCenter)
            info_layout.addWidget(rgb_label)

        if self.color_lab:
            lab_text = ColorConverter.format_lab(self.color_lab)
            lab_label = QLabel(lab_text)
            lab_label.setStyleSheet("font-size: 12px; color: #27ae60; font-weight: bold;")
            lab_label.setAlignment(Qt.AlignCenter)
            info_layout.addWidget(lab_label)

        info_widget.setLayout(info_layout)
        layout.addWidget(info_widget)
        self.setLayout(layout)

    def update_color_display(self):
        if self.color_rgb:
            r, g, b = [max(0, min(255, int(c))) for c in self.color_rgb[:3]]
            color_image = QImage(90, 90, QImage.Format_RGB32)
            color_image.fill(QColor(r, g, b).rgb())
            pixmap = QPixmap.fromImage(color_image)
            painter = QPainter(pixmap)
            painter.setPen(QPen(QColor(0, 0, 0, 50), 1))
            painter.drawRect(0, 0, 89, 89)
            painter.end()
            self.color_block.setPixmap(pixmap)


class SolutionDisplayWidget(QWidget):
    """解决方案显示控件 - 简化版"""
    def __init__(self, solution_data, parent=None):
        super().__init__(parent)
        self.solution_data = solution_data
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)

        solution_type = self.solution_data.get('solution_type', 'unknown')
        title_text = ""
        color = "#3498db"
        if solution_type == 'cluster_best':
            cluster_id = self.solution_data.get('cluster_id', -1)
            title_text = f"Cluster {cluster_id} Best"
            color = "#9b59b6"
        elif solution_type == 'global_best':
            title_text = "Global Best"
            color = "#2ecc71"
        elif solution_type == 'max_deltaED':
            title_text = "Max ΔED"
            color = "#e74c3c"
        else:
            title_text = "Solution"

        title_label = QLabel(title_text)
        title_label.setStyleSheet(f"font-weight: bold; font-size: 15px; color: {color};")
        layout.addWidget(title_label)

        metrics_layout = QGridLayout()
        metrics_layout.setSpacing(5)

        deltaE = self.solution_data.get('deltaE', 0)
        deltaE_label = QLabel(f"ΔE: {deltaE:.3f}")
        deltaE_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        metrics_layout.addWidget(deltaE_label, 0, 0)

        deltaED = self.solution_data.get('deltaED', 0)
        deltaED_label = QLabel(f"ΔED: {deltaED:.3f}")
        deltaED_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        metrics_layout.addWidget(deltaED_label, 0, 1)

        thickness = self.solution_data.get('thickness', [])
        if thickness:
            thickness_text = ColorConverter.format_thickness(thickness)
            thickness_label = QLabel(f"Thickness: {thickness_text}")
            thickness_label.setStyleSheet("font-size: 13px; color: #7f8c8d;")
            metrics_layout.addWidget(thickness_label, 1, 0, 1, 2)

        layout.addLayout(metrics_layout)

        colors_layout = QHBoxLayout()
        colors_layout.setSpacing(10)

        amorphous_lab = self.solution_data.get('pred_lab_amorphous')
        if amorphous_lab:
            amorphous_data = {'lab': amorphous_lab}
            amorphous_widget = ColorDisplayWidget("Amorphous", amorphous_data)
            colors_layout.addWidget(amorphous_widget)

        arrow_label = QLabel("→")
        arrow_label.setStyleSheet("font-size: 20px; font-weight: bold; padding: 0 5px; color: #7f8c8d;")
        colors_layout.addWidget(arrow_label)

        crystalline_lab = self.solution_data.get('pred_lab_crystalline')
        if crystalline_lab:
            crystalline_data = {'lab': crystalline_lab}
            crystalline_widget = ColorDisplayWidget("Crystalline", crystalline_data)
            colors_layout.addWidget(crystalline_widget)

        layout.addLayout(colors_layout)

        if solution_type == 'cluster_best':
            cluster_size = self.solution_data.get('cluster_size', 1)
            size_label = QLabel(f"Size: {cluster_size}")
            size_label.setStyleSheet("font-size: 12px; color: #666; background-color: #f0f0f0; padding: 3px; border-radius: 3px;")
            layout.addWidget(size_label)

        self.setLayout(layout)
        self.setStyleSheet(f"border: 2px solid {color}; border-radius: 8px; background-color: #f9f9f9;")
        self.setMinimumWidth(220)


class DynamicResultsWindow(QWidget):
    """动态设计结果窗口 - 大字体、两列布局、无边框卡片、自动+1、窗口自适应"""
    def __init__(self, all_solutions, parent=None):
        super().__init__()
        self.parent_window = parent
        self.all_solutions = all_solutions
        self.selected_solutions = []
        self.solutions_by_color = self.group_solutions_by_color(all_solutions)
        self.setWindowTitle("Dynamic Design Results")
        self.init_ui()

    def group_solutions_by_color(self, solutions):
        solutions_by_color = {}
        for solution in solutions:
            color_idx = solution.get('color_index', 0)
            if color_idx not in solutions_by_color:
                solutions_by_color[color_idx] = []
            solutions_by_color[color_idx].append(solution)
        return solutions_by_color

    def init_ui(self):
        # ========== 全局字体统一（仿 Step1Window） ==========
        self.setStyleSheet("""
            QLabel {
                font-size: 20px;
            }
            QPushButton {
                font-size: 20px;
                padding: 10px 16px;
                border-radius: 6px;
            }
            QTabWidget::tab {
                font-size: 20px;
                padding: 8px 20px;
            }
            QTabWidget::pane {
                border: 1px solid #dee2e6;
                border-radius: 5px;
                background-color: white;
            }
            QTabBar::tab:selected {
                background-color: white;
                border-bottom: 3px solid #3498db;
            }
            QScrollArea {
                border: none;
                background-color: transparent;
            }
        """)

        # 主布局
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # ----- 标题（大号，蓝色背景）-----
        title_label = QLabel("Dynamic Design Results")
        title_label.setStyleSheet("""
            font-size: 32px;
            font-weight: bold;
            padding: 12px;
            background-color: #3498db;
            color: white;
            border-radius: 8px;
        """)
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # ----- 选项卡区域 -----
        self.tab_widget = QTabWidget()
        for color_idx in sorted(self.solutions_by_color.keys()):
            color_solutions = self.solutions_by_color[color_idx]
            color_tab = self.create_color_tab(color_idx, color_solutions)
            self.tab_widget.addTab(color_tab, f"Color {color_idx + 1}")
        layout.addWidget(self.tab_widget)

        # ----- 底部控制栏 -----
        control_layout = QHBoxLayout()
        control_layout.setSpacing(15)

        self.selection_stats = QLabel("Selected: 0 / 0")
        self.selection_stats.setStyleSheet("""
            font-size: 20px;
            font-weight: bold;
            padding: 8px 16px;
            background-color: #e8f4fd;
            border-radius: 6px;
        """)
        control_layout.addWidget(self.selection_stats)

        control_layout.addStretch()

        select_all_btn = QPushButton("Select All Global Best")
        select_all_btn.setStyleSheet("background-color: #2ecc71; color: white;")
        select_all_btn.clicked.connect(self.select_all_global_best)
        control_layout.addWidget(select_all_btn)

        clear_all_btn = QPushButton("Clear All")
        clear_all_btn.setStyleSheet("background-color: #e74c3c; color: white;")
        clear_all_btn.clicked.connect(self.clear_all_selections)
        control_layout.addWidget(clear_all_btn)

        layout.addLayout(control_layout)

        # ----- 确认按钮（大号）-----
        confirm_btn = QPushButton("Confirm Selection")
        confirm_btn.setStyleSheet("""
            font-size: 24px;
            font-weight: bold;
            padding: 14px;
            background-color: #2ecc71;
            color: white;
            border-radius: 8px;
        """)
        confirm_btn.clicked.connect(self.confirm_selection)
        layout.addWidget(confirm_btn)

        self.setLayout(layout)

        # 窗口自适应
        self.adjustSize()
        self.setMinimumSize(1000, 600)

    def create_color_tab(self, color_idx, solutions):
        """为单个颜色创建选项卡页面"""
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)

        # ----- 目标颜色信息条（紧凑但信息完整）-----
        target_rgb = solutions[0].get('target_rgb', [128, 128, 128])
        target_lab = ColorConverter.rgb_to_lab(target_rgb)

        target_widget = QWidget()
        target_widget.setStyleSheet("""
            border: 1px solid #3498db;
            border-radius: 8px;
            background-color: #f0f8ff;
        """)
        target_layout = QHBoxLayout(target_widget)
        target_layout.setContentsMargins(12, 8, 12, 8)
        target_layout.setSpacing(15)

        color_block = QLabel()
        color_block.setFixedSize(50, 50)
        color_block.setStyleSheet(f"""
            background-color: rgb({target_rgb[0]},{target_rgb[1]},{target_rgb[2]});
            border: 1px solid gray;
            border-radius: 6px;
        """)
        target_layout.addWidget(color_block)

        text_label = QLabel(
            f"Target Color {color_idx + 1}  |  "
            f"RGB({target_rgb[0]}, {target_rgb[1]}, {target_rgb[2]})  |  "
            f"Lab({target_lab[0]:.1f}, {target_lab[1]:.1f}, {target_lab[2]:.1f})"
        )
        text_label.setStyleSheet("font-size: 22px; font-weight: bold; color: #2c3e50;")
        target_layout.addWidget(text_label)
        target_layout.addStretch()
        layout.addWidget(target_widget)

        # ----- 解决方案区域标题 -----
        solutions_label = QLabel("Crystalline Solutions (click to select):")
        solutions_label.setStyleSheet("font-size: 24px; font-weight: bold; margin-top: 5px;")
        layout.addWidget(solutions_label)

        # ----- 可滚动区域（网格布局，每行两列）-----
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(400)
        container = QWidget()
        grid_layout = QGridLayout(container)
        grid_layout.setSpacing(20)
        grid_layout.setContentsMargins(5, 5, 5, 5)

        self.solution_cards = {}
        self.selection_state = {}

        row, col = 0, 0
        max_cols = 2

        for sol in solutions:
            card = self.create_solution_card(sol, color_idx)
            key = (color_idx, sol.get('solution_type'), sol.get('cluster_id', -1))
            self.solution_cards[key] = card
            self.selection_state[key] = False
            card.mousePressEvent = lambda event, k=key: self.toggle_solution_selection(k)

            grid_layout.addWidget(card, row, col)
            col += 1
            if col >= max_cols:
                col = 0
                row += 1

        container.setLayout(grid_layout)
        scroll.setWidget(container)
        layout.addWidget(scroll)

        # ----- 当前选中摘要 -----
        self.current_selection_label = QLabel("Selected: None")
        self.current_selection_label.setObjectName("current_selection_label")
        self.current_selection_label.setStyleSheet("font-size: 20px; color: #27ae60; padding: 5px;")
        layout.addWidget(self.current_selection_label)

        tab.setLayout(layout)
        return tab

    def create_solution_card(self, solution, color_idx):
        """创建无边框、大色块的解决方案卡片"""
        card = QFrame()
        card.setFrameShape(QFrame.NoFrame)
        card.setStyleSheet("""
            QFrame {
                background-color: white;
                border: none;
            }
            QFrame:hover {
                background-color: #f5f9ff;
            }
        """)

        layout = QVBoxLayout(card)
        layout.setSpacing(12)
        layout.setContentsMargins(15, 15, 15, 15)

        sol_type = solution.get('solution_type', 'unknown')
        cluster_id = solution.get('cluster_id', -1)

        # 标题（cluster_id 自动 +1）
        if sol_type == 'cluster_best':
            title = f"Cluster {cluster_id + 1} Best"
            color = "#9b59b6"
        elif sol_type == 'global_best':
            title = "Global Best"
            color = "#2ecc71"
        elif sol_type == 'max_deltaED':
            title = "Max ΔED"
            color = "#e74c3c"
        else:
            title = "Solution"
            color = "#3498db"

        title_label = QLabel(title)
        title_label.setStyleSheet(f"font-size: 24px; font-weight: bold; color: {color};")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # 指标
        deltae = solution.get('deltaE', 0)
        deltaed = solution.get('deltaED', 0)
        metrics = QLabel(f"ΔE: {deltae:.3f}  |  ΔED: {deltaed:.3f}")
        metrics.setStyleSheet("font-size: 22px; font-weight: bold; color: #333;")
        metrics.setAlignment(Qt.AlignCenter)
        layout.addWidget(metrics)

        # 厚度 - 更明显（字体 22px，深橙色）
        thickness = solution.get('thickness', [])
        if thickness:
            thick_str = ColorConverter.format_thickness(thickness)
            thick_label = QLabel(f"Thickness: {thick_str}")
            thick_label.setStyleSheet("font-size: 22px; color: #e67e22; font-weight: bold;")
            thick_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(thick_label)

        # 颜色对比（非晶态→晶态）—— 色块 80×80
        colors_layout = QHBoxLayout()
        colors_layout.setSpacing(15)

        amorphous_lab = solution.get('pred_lab_amorphous', [50, 0, 0])
        crystalline_lab = solution.get('pred_lab_crystalline', [50, 0, 0])
        amorphous_rgb = ColorConverter.lab_to_rgb(amorphous_lab)
        crystalline_rgb = ColorConverter.lab_to_rgb(crystalline_lab)

        def create_color_block(rgb, size=80):
            block = QLabel()
            block.setFixedSize(size, size)
            block.setStyleSheet(f"""
                background-color: rgb({rgb[0]},{rgb[1]},{rgb[2]});
                border: 1px solid #aaa;
                border-radius: 8px;
            """)
            return block

        colors_layout.addStretch()
        colors_layout.addWidget(create_color_block(amorphous_rgb, 80))
        arrow = QLabel("→")
        arrow.setStyleSheet("font-size: 28px; font-weight: bold; color: #7f8c8d;")
        colors_layout.addWidget(arrow)
        colors_layout.addWidget(create_color_block(crystalline_rgb, 80))
        colors_layout.addStretch()
        layout.addLayout(colors_layout)

        return card

    def toggle_solution_selection(self, key):
        """点击卡片时切换选中状态"""
        color_idx, sol_type, cluster_id = key

        # 清除同一颜色下其他选中状态
        for k in list(self.selection_state.keys()):
            if k[0] == color_idx and self.selection_state[k]:
                self.selection_state[k] = False
                self.update_card_style(k)

        self.selection_state[key] = True
        self.update_card_style(key)

        # 更新 selected_solutions 列表
        self.selected_solutions = []
        for k, selected in self.selection_state.items():
            if selected:
                for sol in self.all_solutions:
                    if (sol.get('color_index') == k[0] and
                        sol.get('solution_type') == k[1] and
                        sol.get('cluster_id', -1) == k[2]):
                        self.selected_solutions.append(sol)
                        break

        self.update_current_selection_label(color_idx)
        self.update_selection_stats()

    def update_card_style(self, key):
        card = self.solution_cards.get(key)
        if card:
            if self.selection_state.get(key, False):
                card.setStyleSheet("""
                    QFrame {
                        background-color: #e8f8f5;
                        border: none;
                    }
                """)
            else:
                card.setStyleSheet("""
                    QFrame {
                        background-color: white;
                        border: none;
                    }
                    QFrame:hover {
                        background-color: #f5f9ff;
                    }
                """)

    def update_current_selection_label(self, color_idx):
        current_tab = self.tab_widget.currentWidget()
        if not current_tab:
            return
        for child in current_tab.findChildren(QLabel):
            if child.objectName() == "current_selection_label":
                selected = self.get_selected_solution_for_color(color_idx)
                if selected:
                    st = selected.get('solution_type')
                    cid = selected.get('cluster_id', -1)
                    if st == 'cluster_best':
                        text = f"Selected: Cluster {cid + 1} Best"
                    elif st == 'global_best':
                        text = "Selected: Global Best"
                    elif st == 'max_deltaED':
                        text = "Selected: Max ΔED"
                    else:
                        text = "Selected: Solution"
                else:
                    text = "Selected: None"
                child.setText(text)
                break

    def get_selected_solution_for_color(self, color_idx):
        for sol in self.selected_solutions:
            if sol.get('color_index') == color_idx:
                return sol
        return None

    def select_all_global_best(self):
        """为每个颜色选择 Global Best（参考初始版实现）"""
        # 清空所有选中状态
        self.selected_solutions.clear()
        for key in self.selection_state:
            self.selection_state[key] = False
            self.update_card_style(key)

        # 遍历每个颜色，找到最优解
        for color_idx, solutions in self.solutions_by_color.items():
            # 优先找 global_best 类型
            global_best = None
            for sol in solutions:
                if sol.get('solution_type') == 'global_best':
                    global_best = sol
                    break
            # 如果没有，选 deltaE 最小的
            if not global_best and solutions:
                global_best = min(solutions, key=lambda x: x.get('deltaE', float('inf')))

            if global_best:
                key = (color_idx, global_best.get('solution_type'), global_best.get('cluster_id', -1))
                if key in self.solution_cards:
                    self.selection_state[key] = True
                    self.update_card_style(key)
                    self.selected_solutions.append(global_best)

        # 更新所有界面元素
        self.update_selection_stats()
        # 刷新当前选项卡的摘要
        current_idx = self.tab_widget.currentIndex()
        if current_idx >= 0:
            self.update_current_selection_label(current_idx)

        # 提示操作完成（与初始版一致）
        QMessageBox.information(self, "Info", "Global best selected for all colors")

    def clear_all_selections(self):
        """清空所有选中"""
        self.selected_solutions.clear()
        for key in self.selection_state:
            self.selection_state[key] = False
            self.update_card_style(key)
        self.update_selection_stats()
        current_tab = self.tab_widget.currentWidget()
        if current_tab:
            for child in current_tab.findChildren(QLabel):
                if child.objectName() == "current_selection_label":
                    child.setText("Selected: None")
                    break
        QMessageBox.information(self, "Info", "All selections cleared")

    def update_selection_stats(self):
        selected_count = len(self.selected_solutions)
        total_colors = len(self.solutions_by_color)
        self.selection_stats.setText(f"Selected: {selected_count} / {total_colors}")

    def confirm_selection(self):
        try:
            if not self.selected_solutions:
                QMessageBox.warning(self, "Warning", "Please select at least one solution")
                return

            if len(self.selected_solutions) < len(self.solutions_by_color):
                reply = QMessageBox.question(
                    self, "Confirm",
                    f"Selected {len(self.selected_solutions)} / {len(self.solutions_by_color)} colors. Continue?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply == QMessageBox.No:
                    return

            simplified_designs = []
            solutions_by_color = {}
            for sol in self.selected_solutions:
                color_idx = sol.get('color_index', 0)
                if color_idx not in solutions_by_color:
                    solutions_by_color[color_idx] = []
                solutions_by_color[color_idx].append(sol)

            for color_idx in sorted(solutions_by_color.keys()):
                color_solutions = solutions_by_color[color_idx]
                if color_solutions:
                    simplified_solutions = []
                    for sol in color_solutions:
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
                    simplified_designs.append({
                        'color_index': color_idx,
                        'target_rgb': color_solutions[0].get('target_rgb', [128, 128, 128]),
                        'solutions': simplified_solutions
                    })

            if self.parent_window:
                self.parent_window.dynamic_results_confirmed(simplified_designs)
                self.close()
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Confirmation failed: {str(e)}")


class StaticResultsWindow(QWidget):
    """静态设计结果窗口 - 统一字体、无边框卡片、大色块、厚度醒目、窗口自适应"""
    def __init__(self, designs, parent=None):
        super().__init__()
        self.parent_window = parent
        self.designs = designs  # 每个元素是一个颜色的设计结果（包含 global_best 和 max_deltaED 等）
        self.setWindowTitle("Static Design Results")
        self.init_ui()

    def init_ui(self):
        # ========== 全局字体统一（仿 Step1Window 与 DynamicResultsWindow） ==========
        self.setStyleSheet("""
            QLabel {
                font-size: 20px;
            }
            QPushButton {
                font-size: 20px;
                padding: 10px 16px;
                border-radius: 6px;
            }
            QTabWidget::tab {
                font-size: 20px;
                padding: 8px 20px;
            }
            QTabWidget::pane {
                border: 1px solid #dee2e6;
                border-radius: 5px;
                background-color: white;
            }
            QTabBar::tab:selected {
                background-color: white;
                border-bottom: 3px solid #2ecc71;
            }
            QGroupBox {
                font-size: 20px;
                font-weight: bold;
                border: 2px solid #2ecc71;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px;
            }
            QTableWidget {
                font-size: 18px;
                border: none;
            }
            QHeaderView::section {
                font-size: 18px;
                font-weight: bold;
                background-color: #f0f0f0;
                padding: 6px;
            }
        """)

        # 主布局
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # ----- 标题（大号，绿色背景）-----
        title_label = QLabel("Static Design Results")
        title_label.setStyleSheet("""
            font-size: 32px;
            font-weight: bold;
            padding: 12px;
            background-color: #2ecc71;
            color: white;
            border-radius: 8px;
        """)
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # ----- 选项卡区域 -----
        self.tab_widget = QTabWidget()
        sorted_designs = sorted(self.designs, key=lambda x: x.get('color_index', 0))
        for i, design in enumerate(sorted_designs):
            color_idx = design.get('color_index', i)
            color_tab = self.create_color_tab(color_idx, design)
            self.tab_widget.addTab(color_tab, f"Color {color_idx + 1}")
        layout.addWidget(self.tab_widget)

        # ----- 统计信息（组框）-----
        stats_text = self.calculate_statistics()
        stats_group = QGroupBox("Statistics")
        stats_group.setStyleSheet("""
            QGroupBox {
                font-size: 22px;
                font-weight: bold;
                border: 2px solid #3498db;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 15px;
            }
        """)
        stats_layout = QVBoxLayout()
        stats_label = QLabel(stats_text)
        stats_label.setStyleSheet("""
            font-size: 20px;
            padding: 12px;
            background-color: #e8f4fd;
            border-radius: 6px;
        """)
        stats_label.setWordWrap(True)
        stats_layout.addWidget(stats_label)
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)

        # ----- 确认按钮（大号）-----
        confirm_btn = QPushButton("Confirm Design Results")
        confirm_btn.setStyleSheet("""
            font-size: 24px;
            font-weight: bold;
            padding: 14px;
            background-color: #2ecc71;
            color: white;
            border-radius: 8px;
        """)
        confirm_btn.clicked.connect(self.confirm_designs)
        layout.addWidget(confirm_btn)

        self.setLayout(layout)

        # 窗口自适应
        self.adjustSize()
        self.setMinimumSize(1000, 600)

    def create_color_tab(self, color_idx, design):
        """为单个颜色创建选项卡页面"""
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)

        # ----- 目标颜色信息条（紧凑但信息完整）-----
        target_rgb = design.get('target_rgb', [128, 128, 128])
        target_lab = ColorConverter.rgb_to_lab(target_rgb)

        target_widget = QWidget()
        target_widget.setStyleSheet("""
            border: 1px solid #3498db;
            border-radius: 8px;
            background-color: #f0f8ff;
        """)
        target_layout = QHBoxLayout(target_widget)
        target_layout.setContentsMargins(12, 8, 12, 8)
        target_layout.setSpacing(15)

        color_block = QLabel()
        color_block.setFixedSize(50, 50)
        color_block.setStyleSheet(f"""
            background-color: rgb({target_rgb[0]},{target_rgb[1]},{target_rgb[2]});
            border: 1px solid gray;
            border-radius: 6px;
        """)
        target_layout.addWidget(color_block)

        text_label = QLabel(
            f"Target Color {color_idx + 1}  |  "
            f"RGB({target_rgb[0]}, {target_rgb[1]}, {target_rgb[2]})  |  "
            f"Lab({target_lab[0]:.1f}, {target_lab[1]:.1f}, {target_lab[2]:.1f})"
        )
        text_label.setStyleSheet("font-size: 22px; font-weight: bold; color: #2c3e50;")
        target_layout.addWidget(text_label)
        target_layout.addStretch()
        layout.addWidget(target_widget)

        # ----- 解决方案卡片区域（水平排列）-----
        solutions_group = QGroupBox("Design Results")
        solutions_group.setStyleSheet("""
            QGroupBox {
                font-size: 22px;
                font-weight: bold;
                border: 2px solid #2ecc71;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 15px;
            }
        """)
        solutions_layout = QHBoxLayout()
        solutions_layout.setSpacing(20)

        # 获取设计结果
        if 'design_result' in design:
            solutions_data = design['design_result'].get('solutions', {})
        else:
            solutions_data = {'global_best': design}

        # Global Best 卡片
        global_best = solutions_data.get('global_best')
        if global_best:
            global_card = self.create_result_card("Global Best", global_best, "#2ecc71")
            solutions_layout.addWidget(global_card)

        # Max ΔED 卡片
        max_deltaED = solutions_data.get('max_deltaED')
        if max_deltaED:
            max_card = self.create_result_card("Max ΔED", max_deltaED, "#e74c3c")
            solutions_layout.addWidget(max_card)

        # 如果缺少 max_deltaED，显示占位符（样式统一）
        if not max_deltaED:
            placeholder = QLabel("No max ΔED solution")
            placeholder.setStyleSheet("""
                font-size: 20px;
                color: #666;
                padding: 30px;
                border: 2px dashed #ccc;
                border-radius: 8px;
                background-color: #f9f9f9;
            """)
            placeholder.setAlignment(Qt.AlignCenter)
            solutions_layout.addWidget(placeholder)

        solutions_layout.addStretch()
        solutions_group.setLayout(solutions_layout)
        layout.addWidget(solutions_group)

        # ----- 参数表格（字体加大）-----
        detail_label = QLabel("Parameters:")
        detail_label.setStyleSheet("font-size: 22px; font-weight: bold; padding: 5px 0;")
        layout.addWidget(detail_label)

        table = QTableWidget()
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(["Parameter", "Value", "Unit", "Description"])
        table.setAlternatingRowColors(True)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setStyleSheet("""
            QTableWidget {
                font-size: 18px;
                border: 1px solid #dee2e6;
            }
            QTableWidget::item {
                padding: 8px;
            }
        """)
        self.populate_design_table(table, design)
        table.setMaximumHeight(200)
        layout.addWidget(table)

        layout.addStretch()
        tab.setLayout(layout)
        return tab

    def create_result_card(self, title, solution, color):
        """创建无边框、大色块、厚度突出的卡片（与动态窗口风格一致）"""
        card = QFrame()
        card.setFrameShape(QFrame.NoFrame)
        card.setStyleSheet("""
            QFrame {
                background-color: white;
                border: none;
            }
            QFrame:hover {
                background-color: #f5f9ff;
            }
        """)

        layout = QVBoxLayout(card)
        layout.setSpacing(12)
        layout.setContentsMargins(15, 15, 15, 15)

        # 标题
        title_label = QLabel(title)
        title_label.setStyleSheet(f"font-size: 24px; font-weight: bold; color: {color};")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # 指标
        deltae = solution.get('deltaE', 0)
        deltaed = solution.get('deltaED', 0)
        metrics = QLabel(f"ΔE: {deltae:.3f}  |  ΔED: {deltaed:.3f}")
        metrics.setStyleSheet("font-size: 22px; font-weight: bold; color: #333;")
        metrics.setAlignment(Qt.AlignCenter)
        layout.addWidget(metrics)

        # 厚度 - 更明显
        thickness = solution.get('thickness', [])
        if thickness:
            thick_str = ColorConverter.format_thickness(thickness)
            thick_label = QLabel(f"Thickness: {thick_str}")
            thick_label.setStyleSheet("font-size: 22px; color: #e67e22; font-weight: bold;")
            thick_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(thick_label)

        # 颜色对比（非晶态→晶态）—— 大色块
        colors_layout = QHBoxLayout()
        colors_layout.setSpacing(15)

        amorphous_lab = solution.get('pred_lab_amorphous', [50, 0, 0])
        crystalline_lab = solution.get('pred_lab_crystalline', [50, 0, 0])
        amorphous_rgb = ColorConverter.lab_to_rgb(amorphous_lab)
        crystalline_rgb = ColorConverter.lab_to_rgb(crystalline_lab)

        def create_color_block(rgb, size=80):
            block = QLabel()
            block.setFixedSize(size, size)
            block.setStyleSheet(f"""
                background-color: rgb({rgb[0]},{rgb[1]},{rgb[2]});
                border: 1px solid #aaa;
                border-radius: 8px;
            """)
            return block

        colors_layout.addStretch()
        colors_layout.addWidget(create_color_block(amorphous_rgb, 80))
        arrow = QLabel("→")
        arrow.setStyleSheet("font-size: 28px; font-weight: bold; color: #7f8c8d;")
        colors_layout.addWidget(arrow)
        colors_layout.addWidget(create_color_block(crystalline_rgb, 80))
        colors_layout.addStretch()
        layout.addLayout(colors_layout)

        return card

    def populate_design_table(self, table, design):
        """填充参数表格，保留原有内容，字体已整体放大"""
        rows = []
        rows.append(["ΔE (amorphous-target)", f"{design.get('deltaE', 0):.3f}", "", ""])
        rows.append(["ΔED (crystalline-amorphous)", f"{design.get('deltaED', 0):.3f}", "", ""])

        thickness = design.get('thickness', [])
        if thickness:
            thickness_str = ColorConverter.format_thickness(thickness)
            rows.append(["Thickness", thickness_str, "nm", ""])

        target_rgb = design.get('target_rgb', [])
        if target_rgb:
            rows.append(["Target RGB", f"{target_rgb[0]}, {target_rgb[1]}, {target_rgb[2]}", "", ""])

        amorphous_rgb = design.get('pred_rgb_amorphous', [])
        if amorphous_rgb:
            rows.append(["Amorphous RGB", f"{amorphous_rgb[0]}, {amorphous_rgb[1]}, {amorphous_rgb[2]}", "", ""])

        crystalline_rgb = design.get('pred_rgb_crystalline', [])
        if crystalline_rgb:
            rows.append(["Crystalline RGB", f"{crystalline_rgb[0]}, {crystalline_rgb[1]}, {crystalline_rgb[2]}", "", ""])

        solution_type = design.get('solution_type', 'global_best')
        rows.append(["Solution type", solution_type, "", ""])

        table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            for j, value in enumerate(row):
                item = QTableWidgetItem(str(value))
                table.setItem(i, j, item)
            table.setRowHeight(i, 36)  # 行高加大以适应字体

    def calculate_statistics(self):
        """计算统计信息（与原逻辑一致，仅调整格式）"""
        if not self.designs:
            return "No data"
        total_colors = len(self.designs)
        deltaE_values = [d.get('deltaE', 0) for d in self.designs]
        deltaED_values = [d.get('deltaED', 0) for d in self.designs]
        avg_deltaE = sum(deltaE_values) / total_colors if total_colors > 0 else 0
        avg_deltaED = sum(deltaED_values) / total_colors if total_colors > 0 else 0
        min_deltaE = min(deltaE_values) if deltaE_values else 0
        max_deltaE = max(deltaE_values) if deltaE_values else 0
        min_deltaED = min(deltaED_values) if deltaED_values else 0
        max_deltaED = max(deltaED_values) if deltaED_values else 0

        stats_text = (
            f"Colors: {total_colors}\n"
            f"ΔE avg: {avg_deltaE:.3f} (min {min_deltaE:.3f}, max {max_deltaE:.3f})\n"
            f"ΔED avg: {avg_deltaED:.3f} (min {min_deltaED:.3f}, max {max_deltaED:.3f})"
        )
        return stats_text

    def confirm_designs(self):
        """确认设计结果"""
        summary = self.calculate_statistics()
        reply = QMessageBox.question(
            self, "Confirm",
            f"{summary}\n\nConfirm these designs?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            if self.parent_window:
                self.parent_window.static_results_confirmed(self.designs)
            self.close()


class ResultWindow(QWidget):
    """最终结果窗口 - 字体统一放大，信息紧凑，光谱分段着色"""
    def __init__(self, design_params, main_window):
        super().__init__()
        self.design_params = design_params
        self.main_window = main_window
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Final Results")
        self.setMinimumSize(900, 600)  # 宽度减小，高度自适应

        # 全局样式 - 按Step1Window字体体系放大
        self.setStyleSheet("""
            QLabel { font-size: 20px; }
            QPushButton { font-size: 22px; padding: 8px 16px; }
            QGroupBox { font-size: 22px; font-weight: bold; margin-top: 12px; }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 6px; }
            QTabWidget::tab { font-size: 20px; padding: 6px 14px; }
        """)

        layout = QVBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(12)

        design_type = self.design_params.get('design_type', 'unknown')
        title_text = "Dynamic Design Results" if design_type == 'dynamic' else "Static Design Results"
        title_label = QLabel(title_text)
        # 标题不使用白色字体，深色文字+浅灰渐变背景
        title_label.setStyleSheet("""
            font-size: 32px; 
            font-weight: bold; 
            padding: 12px; 
            background: linear-gradient(90deg, #e0e0e0, #c0c0c0); 
            color: #2c3e50; 
            border-radius: 8px;
        """)
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        tab_widget = QTabWidget()
        tab_widget.setStyleSheet("""
            QTabWidget::pane { border: 2px solid #dee2e6; border-radius: 6px; background-color: white; }
            QTabBar::tab:selected { background-color: white; border-bottom: 3px solid #3498db; }
        """)

        # 三个选项卡
        summary_tab = self.create_summary_tab()
        tab_widget.addTab(summary_tab, "📊 Summary")

        spectrum_tab = self.create_spectrum_tab()
        tab_widget.addTab(spectrum_tab, "📈 Spectrum")

        pattern_tab = self.create_pattern_tab()
        tab_widget.addTab(pattern_tab, "🖼️ Pattern")

        layout.addWidget(tab_widget)

        # 底部按钮
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        save_btn = QPushButton("Save Results")
        save_btn.setStyleSheet("background-color: #3498db; color: white; font-weight: bold; border-radius: 8px; padding: 12px 24px; font-size: 24px;")
        save_btn.clicked.connect(self.save_results)
        button_layout.addWidget(save_btn)

        restart_btn = QPushButton("Restart")
        restart_btn.setStyleSheet("background-color: #f39c12; color: white; font-weight: bold; border-radius: 8px; padding: 12px 24px; font-size: 24px;")
        restart_btn.clicked.connect(self.restart_design)
        button_layout.addWidget(restart_btn)

        exit_btn = QPushButton("Exit")
        exit_btn.setStyleSheet("background-color: #e74c3c; color: white; font-weight: bold; border-radius: 8px; padding: 12px 24px; font-size: 24px;")
        exit_btn.clicked.connect(self.close_app)
        button_layout.addWidget(exit_btn)

        layout.addLayout(button_layout)
        self.setLayout(layout)
        self.adjustSize()

    def create_summary_tab(self):
        """摘要选项卡：显示核心统计和颜色对比"""
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        design_details = self.design_params.get('design_details', [])
        design_type = self.design_params.get('design_type', 'unknown')
        color_count = len(design_details)
        solutions_count = self.design_params.get('solutions_count', 0)

        # 统计卡片
        stats_group = QGroupBox("Overall Statistics")
        stats_group.setStyleSheet("font-size: 24px; font-weight: bold;")
        stats_layout = QHBoxLayout()

        avg_deltaE = 0
        avg_deltaED = 0
        if design_details:
            avg_deltaE = sum(d.get('deltaE', 0) for d in design_details) / len(design_details)
            avg_deltaED = sum(d.get('deltaED', 0) for d in design_details) / len(design_details)

        stats_text = (f"Type: {'Dynamic' if design_type == 'dynamic' else 'Static'}\n"
                      f"Colors: {color_count}\n"
                      f"Solutions: {solutions_count}\n"
                      f"Avg ΔE: {avg_deltaE:.3f}\n"
                      f"Avg ΔED: {avg_deltaED:.3f}")
        stats_label = QLabel(stats_text)
        stats_label.setStyleSheet("font-size: 22px; padding: 15px; background-color: #e8f4fd; border-radius: 8px;")
        stats_label.setAlignment(Qt.AlignCenter)
        stats_layout.addWidget(stats_label)
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)

        # 颜色对比概览
        if design_details:
            colors_group = QGroupBox("Color Comparison")
            colors_group.setStyleSheet("font-size: 24px; font-weight: bold;")
            colors_layout = QVBoxLayout()
            for design in design_details:
                color_widget = self.create_compact_color_row(design)
                colors_layout.addWidget(color_widget)
            colors_group.setLayout(colors_layout)
            layout.addWidget(colors_group)

        layout.addStretch()
        tab.setLayout(layout)
        return tab

    def create_compact_color_row(self, design):
        """为每个颜色创建一行紧凑的颜色对比，字体调大，颜色块略缩小以节省宽度"""
        widget = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(3, 3, 3, 3)
        layout.setSpacing(8)

        color_idx = design.get('color_index', 0) + 1
        target_rgb = design.get('target_rgb', [128,128,128])
        amorphous_rgb = design.get('pred_rgb_amorphous', [128,128,128])
        crystalline_rgb = design.get('pred_rgb_crystalline', [128,128,128])
        deltaE = design.get('deltaE', 0)
        deltaED = design.get('deltaED', 0)

        # 颜色块尺寸32x32（原40）
        def create_color_block(rgb, size=32):
            block = QLabel()
            block.setFixedSize(size, size)
            block.setStyleSheet(f"background-color: rgb({rgb[0]},{rgb[1]},{rgb[2]}); border: 2px solid gray; border-radius: 4px;")
            return block

        idx_label = QLabel(f"Color {color_idx}:")
        idx_label.setStyleSheet("font-size: 20px;")
        layout.addWidget(idx_label)
        layout.addWidget(create_color_block(target_rgb))
        layout.addWidget(QLabel("→"))
        layout.addWidget(create_color_block(amorphous_rgb))
        layout.addWidget(QLabel("→"))
        layout.addWidget(create_color_block(crystalline_rgb))
        layout.addSpacing(15)

        delta_label = QLabel(f"ΔE: {deltaE:.3f}  ΔED: {deltaED:.3f}")
        delta_label.setStyleSheet("font-size: 20px;")
        layout.addWidget(delta_label)
        layout.addStretch()

        widget.setLayout(layout)
        return widget

    def create_spectrum_tab(self):
        """光谱选项卡：只显示3-14um，背景分段着色，字体放大"""
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)

        design_details = self.design_params.get('design_details', [])
        if not design_details:
            no_data_label = QLabel("No spectrum data available")
            no_data_label.setStyleSheet("font-size: 24px; color: #7f8c8d; padding: 50px;")
            no_data_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(no_data_label)
            tab.setLayout(layout)
            return tab

        # 获取光谱数据
        spectrum_data = self.design_params.get('spectrum_results', {})
        if not spectrum_data:
            try:
                from core.spectrum_calculation import calculate_all_spectra
                spectrum_data = calculate_all_spectra(
                    self.design_params.get('selected_designs', []),
                    wavelength_range=(3000, 14000, 10)  # 3-14um
                )
                self.design_params['spectrum_results'] = spectrum_data
            except Exception as e:
                print(f"Spectrum calculation failed: {e}")
                spectrum_data = {}

        # 创建 Matplotlib 图表，宽度10英寸（原12），使布局更窄
        fig = Figure(figsize=(10, 6), dpi=100)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)

        # 设置x轴范围3-14um
        ax.set_xlim(3000, 14000)

        # 分段背景着色
        ax.axvspan(3000, 5000, alpha=0.2, color='lightcoral', label='3-5 μm')
        ax.axvspan(5000, 8000, alpha=0.2, color='lightgreen', label='5-8 μm')
        ax.axvspan(8000, 14000, alpha=0.2, color='lightblue', label='8-14 μm')

        if spectrum_data and 'colors' in spectrum_data:
            wavelengths = spectrum_data.get('wavelengths', [])
            colors_data = spectrum_data['colors']
            for color_key, color_data in colors_data.items():
                amorphous_emiss = color_data['amorphous']['emissivity']
                crystalline_emiss = color_data['crystalline']['emissivity']
                if amorphous_emiss and len(amorphous_emiss) > 0:
                    ax.plot(wavelengths, amorphous_emiss[0], label=f"{color_key} Amorphous", linestyle='--')
                if crystalline_emiss and len(crystalline_emiss) > 0:
                    ax.plot(wavelengths, crystalline_emiss[0], label=f"{color_key} Crystalline", linestyle='-')
            # 字体放大
            ax.set_xlabel("Wavelength (nm)", fontsize=16)
            ax.set_ylabel("Emissivity", fontsize=16)
            ax.set_title("Infrared Emissivity Spectrum (3-14 μm)", fontsize=18)
            ax.legend(fontsize=14)
            ax.tick_params(axis='both', labelsize=14)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
        else:
            ax.text(0.5, 0.5, "No spectrum data", ha='center', va='center', transform=ax.transAxes, fontsize=16)

        fig.tight_layout()
        layout.addWidget(canvas)
        tab.setLayout(layout)
        return tab

    def create_pattern_tab(self):
        """图案选项卡：去掉Environment信息，字体放大"""
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)

        amorphous_pattern = self.design_params.get('amorphous_pattern')
        crystalline_pattern = self.design_params.get('crystalline_pattern')

        if amorphous_pattern is None and crystalline_pattern is None:
            no_pattern_widget = QWidget()
            no_pattern_layout = QVBoxLayout()
            no_pattern_layout.setAlignment(Qt.AlignCenter)
            no_pattern_icon = QLabel("🚫")
            no_pattern_icon.setStyleSheet("font-size: 80px;")
            no_pattern_text = QLabel("No camouflage pattern generated")
            no_pattern_text.setStyleSheet("font-size: 24px; color: #7f8c8d;")
            no_pattern_layout.addWidget(no_pattern_icon)
            no_pattern_layout.addWidget(no_pattern_text)
            no_pattern_widget.setLayout(no_pattern_layout)
            layout.addWidget(no_pattern_widget)
            tab.setLayout(layout)
            return tab

        # 使用子选项卡显示非晶态和晶态图案
        pattern_tabs = QTabWidget()

        if amorphous_pattern is not None:
            amorphous_tab = self.create_pattern_display_tab(amorphous_pattern, "Amorphous")
            pattern_tabs.addTab(amorphous_tab, "Amorphous")

        if crystalline_pattern is not None:
            crystalline_tab = self.create_pattern_display_tab(crystalline_pattern, "Crystalline")
            pattern_tabs.addTab(crystalline_tab, "Crystalline")

        layout.addWidget(pattern_tabs)

        # 生成信息 - 去掉Environment
        info_group = QGroupBox("Generation Info")
        info_group.setStyleSheet("font-size: 24px; font-weight: bold;")
        info_layout = QHBoxLayout()
        pattern_size = self.design_params.get('pattern_size', (256, 256))
        size_label = QLabel(f"Size: {pattern_size[0]}×{pattern_size[1]}")
        size_label.setStyleSheet("font-size: 22px;")
        info_layout.addWidget(size_label)
        color_count = self.design_params.get('color_count', 0)
        color_label = QLabel(f"Colors: {color_count}")
        color_label.setStyleSheet("font-size: 22px;")
        info_layout.addWidget(color_label)
        info_layout.addStretch()
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        tab.setLayout(layout)
        return tab

    def create_pattern_display_tab(self, pattern_array, title):
        """创建单个图案显示选项卡，说明字体调大"""
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)

        try:
            height, width, channels = pattern_array.shape
            if channels == 3:
                bytes_per_line = 3 * width
                qimage = QImage(pattern_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
            else:
                bytes_per_line = 4 * width
                qimage = QImage(pattern_array.data, width, height, bytes_per_line, QImage.Format_RGBA8888)
            pixmap = QPixmap.fromImage(qimage)
            pattern_display = QLabel()
            pattern_display.setAlignment(Qt.AlignCenter)
            pattern_display.setStyleSheet("border: 3px solid #bdc3c7; border-radius: 10px;")
            scaled_pixmap = pixmap.scaled(450, 450, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            pattern_display.setPixmap(scaled_pixmap)
            layout.addWidget(pattern_display)
            info_label = QLabel(f"Size: {width}×{height}")
            info_label.setStyleSheet("font-size: 20px; color: #7f8c8d; padding: 8px;")
            layout.addWidget(info_label)
        except Exception as e:
            error_label = QLabel(f"Display error: {str(e)}")
            error_label.setStyleSheet("color: red; font-size: 20px;")
            layout.addWidget(error_label)

        tab.setLayout(layout)
        return tab

    def save_results(self):
        """保存结果（保持原有逻辑）"""
        try:
            from utils.file_handlers import save_design_results
            spectrum_data = self.design_params.get('spectrum_results', {})
            pattern_image = self.design_params.get('amorphous_pattern')
            save_path = save_design_results(
                self.design_params,
                spectrum_data,
                pattern_image,
                "./results"
            )
            if save_path:
                QMessageBox.information(
                    self,
                    "Save Successful",
                    f"Results saved to {list(save_path.values())[0]}",
                    QMessageBox.Ok
                )
            else:
                QMessageBox.warning(self, "Save Failed", "Cannot save results.")
        except Exception as e:
            QMessageBox.critical(self, "Save Failed", f"Error: {str(e)}")

    def restart_design(self):
        reply = QMessageBox.question(
            self,
            "Restart",
            "Restart design? Current results will not be saved.",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.close()
            if self.main_window:
                self.main_window.show()

    def close_app(self):
        reply = QMessageBox.question(
            self,
            "Exit",
            "Exit application?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            QApplication.quit()