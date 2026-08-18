"""
辅助函数
"""

import os
import random
import sys
import numpy as np
from datetime import datetime


def get_default_dir():
    """
    文件对话框的默认目录：项目根目录（main.py 所在目录）

    无论从何处启动，打开/保存对话框都默认定位到项目自身目录，
    避免暴露用户私人路径。
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_image_dir():
    """
    图像选择对话框的默认目录：项目根目录下的 demo_images 文件夹

    专门用于存放演示/输入图像；不存在时自动创建。
    """
    image_dir = os.path.join(get_default_dir(), 'demo_images')
    os.makedirs(image_dir, exist_ok=True)
    return image_dir


def safe_flush_stdout():
    """
    安全地刷新标准输出

    在无控制台环境（如 pythonw 或 detached 进程）中，
    sys.stdout 可能为 None 或其底层句柄已失效，
    直接调用 flush() 会抛出 OSError/ValueError。
    """
    try:
        if sys.stdout is not None:
            sys.stdout.flush()
    except (OSError, ValueError, AttributeError):
        pass


def normalize_color_values(values, color_space):
    """
    将颜色值归一化到标准范围

    输入:
        values: list[float] - 颜色值列表
        color_space: str - 颜色空间名称

    输出:
        normalized_values: list[float] - 归一化后的颜色值
    """
    if color_space == 'RGB':
        return values
    elif color_space == 'Lab':
        normalized = []
        normalized.append(max(0, min(100, values[0])))
        normalized.append(max(-128, min(127, values[1])))
        normalized.append(max(-128, min(127, values[2])))
        return normalized
    elif color_space == 'XYZ':
        normalized = []
        normalized.append(max(0, min(0.95, values[0])))
        normalized.append(max(0, min(1.0, values[1])))
        normalized.append(max(0, min(1.09, values[2])))
        return normalized
    else:
        return values


def calculate_color_distance(color1, color2, color_space='RGB'):
    """
    计算两个颜色之间的距离

    输入:
        color1: tuple/list - 第一个颜色
        color2: tuple/list - 第二个颜色
        color_space: str - 颜色空间

    输出:
        distance: float - 颜色距离
    """
    if len(color1) != 3 or len(color2) != 3:
        return float('inf')

    # 欧几里得距离
    if color_space == 'RGB':
        r1, g1, b1 = color1
        r2, g2, b2 = color2
        return np.sqrt((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2)

    elif color_space == 'Lab':
        L1, a1, b1 = color1
        L2, a2, b2 = color2
        return np.sqrt((L1 - L2) ** 2 + (a1 - a2) ** 2 + (b1 - b2) ** 2)

    else:
        return np.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(color1, color2)))


def generate_unique_filename(base_name, extension='png'):
    """
    生成唯一的文件名

    输入:
        base_name: str - 基础文件名
        extension: str - 文件扩展名

    输出:
        filename: str - 唯一文件名
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_id = random.randint(1000, 9999)

    # 清理基础名称中的非法字符
    safe_base = ''.join(c for c in base_name if c.isalnum() or c in ('_', '-')).strip()
    if not safe_base:
        safe_base = 'design'

    return f"{safe_base}_{timestamp}_{random_id}.{extension.lstrip('.')}"


def rgb_to_hex(rgb):
    """
    将RGB颜色转换为十六进制字符串

    输入:
        rgb: tuple - RGB颜色值

    输出:
        str - 十六进制颜色代码
    """
    if len(rgb) >= 3:
        return '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0]), int(rgb[1]), int(rgb[2])
        )
    return '#000000'


def hex_to_rgb(hex_color):
    """
    将十六进制颜色转换为RGB

    输入:
        hex_color: str - 十六进制颜色代码

    输出:
        tuple - RGB颜色值
    """
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 6:
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (0, 0, 0)