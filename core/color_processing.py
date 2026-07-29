"""
颜色处理功能 - 修改支持可变颜色数量输出
"""

import random
import numpy as np
try:
    from colour import XYZ_to_Lab, Lab_to_XYZ, RGB_to_XYZ, XYZ_to_RGB
    from colour.models import RGB_COLOURSPACE_sRGB
    COLOUR_SCIENCE_AVAILABLE = True
except ImportError:
    COLOUR_SCIENCE_AVAILABLE = False


def extract_dominant_colors(image_path, n_colors=None, color_space='RGB'):
    """
    从图像中提取主要颜色及其权重

    输入:
        image_path: str - 图像文件路径
        n_colors: int - 要提取的颜色数量（None表示由聚类算法决定）
        color_space: str - 颜色空间 ('RGB', 'Lab', 'XYZ')

    输出:
        tuple: (colors, weights)
            colors: list of RGB tuples [(R,G,B), ...] - 颜色列表
            weights: list of float - 对应颜色的权重（占比或代表性权重）
    """
    from Cluster_extraction.extract_color_features import ColorFeatureExtractor
    extractor = ColorFeatureExtractor(
        n_colors=4,  # 提取4种颜色
        show_progress=True  # 显示进度信息
    )
    result = extractor.extract_colors(image_path)[0]  # 单图像结果，格式为 [colors, weights]
    colors = result[0]
    weights = result[1]
    # print(colors)
    # print(weights)
    # colors=[(113, 142, 74), (76, 105, 45), (145, 177, 94), (220, 222, 188)]
    # weights=[0.5700216293334961, 0.29250526428222656, 0.11685466766357422, 0.020618438720703125]
    return colors, weights


def convert_color_space(color, from_space, to_space):
    """
    颜色空间转换

    输入:
        color: tuple/list - 颜色值
        from_space: str - 原颜色空间
        to_space: str - 目标颜色空间

    输出:
        tuple - 转换后的颜色值
    """
    if from_space == to_space:
        return tuple(color)

    if not COLOUR_SCIENCE_AVAILABLE:
        # 如果没有colour-science库，使用简化转换
        return _simplified_color_conversion(color, from_space, to_space)

    # 使用colour-science库进行转换
    try:
        if from_space == 'RGB' and to_space == 'Lab':
            # RGB -> XYZ -> Lab
            xyz = RGB_to_XYZ(color, RGB_COLOURSPACE_sRGB)
            lab = XYZ_to_Lab(xyz)
            return tuple(lab)
        elif from_space == 'Lab' and to_space == 'RGB':
            # Lab -> XYZ -> RGB
            xyz = Lab_to_XYZ(color)
            rgb = XYZ_to_RGB(xyz, RGB_COLOURSPACE_sRGB)
            return tuple(int(max(0, min(255, c * 255))) for c in rgb)
        elif from_space == 'RGB' and to_space == 'XYZ':
            xyz = RGB_to_XYZ(color, RGB_COLOURSPACE_sRGB)
            return tuple(xyz)
        elif from_space == 'XYZ' and to_space == 'RGB':
            rgb = XYZ_to_RGB(color, RGB_COLOURSPACE_sRGB)
            return tuple(int(max(0, min(255, c * 255))) for c in rgb)
        else:
            # 其他转换简化为返回原值
            return tuple(color)
    except Exception:
        return _simplified_color_conversion(color, from_space, to_space)


def _simplified_color_conversion(color, from_space, to_space):
    """简化的颜色空间转换"""
    if from_space == 'RGB' and to_space == 'Lab':
        # 简化的RGB到Lab转换
        r, g, b = color
        x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
        y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
        z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b

        # 简化的XYZ到Lab转换
        x_ref, y_ref, z_ref = 95.047, 100.0, 108.883
        x = x / x_ref
        y = y / y_ref
        z = z / z_ref

        epsilon = 0.008856
        kappa = 903.3

        fx = x ** (1/3) if x > epsilon else (kappa * x + 16) / 116
        fy = y ** (1/3) if y > epsilon else (kappa * y + 16) / 116
        fz = z ** (1/3) if z > epsilon else (kappa * z + 16) / 116

        L = 116 * fy - 16
        a = 500 * (fx - fy)
        b = 200 * (fy - fz)

        return (L, a, b)

    elif from_space == 'Lab' and to_space == 'RGB':
        # 简化的Lab到RGB转换
        L, a, b = color

        # 简化的Lab到XYZ转换
        fy = (L + 16) / 116
        fx = a / 500 + fy
        fz = fy - b / 200

        x_ref, y_ref, z_ref = 95.047, 100.0, 108.883
        epsilon = 0.008856
        kappa = 903.3

        xr = fx ** 3 if fx ** 3 > epsilon else (116 * fx - 16) / kappa
        yr = ((L + 16) / 116) ** 3 if L > kappa * epsilon else L / kappa
        zr = fz ** 3 if fz ** 3 > epsilon else (116 * fz - 16) / kappa

        x = xr * x_ref
        y = yr * y_ref
        z = zr * z_ref

        # 简化的XYZ到RGB转换
        r = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z
        g = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z
        b = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z

        # 限制到0-255范围
        r = max(0, min(255, int(r)))
        g = max(0, min(255, int(g)))
        b = max(0, min(255, int(b)))

        return (r, g, b)

    else:
        # 其他转换简化为返回原值
        return tuple(color)


def validate_color_values(values, color_space):
    """
    验证颜色值是否在有效范围内

    输入:
        values: list - 颜色值列表
        color_space: str - 颜色空间

    输出:
        tuple - (是否有效, 错误信息)
    """
    if len(values) != 3:
        return False, "必须输入3个颜色分量"

    # 定义各颜色空间的范围
    ranges = {
        'RGB': [(0, 255), (0, 255), (0, 255)],
        'Lab': [(0, 100), (-128, 127), (-128, 127)],
        'XYZ': [(0, 0.95), (0, 1.0), (0, 1.09)],
    }

    if color_space not in ranges:
        return False, f"不支持的颜色空间: {color_space}"

    # 验证每个分量
    for i, (value, (min_val, max_val)) in enumerate(zip(values, ranges[color_space])):
        if not (min_val <= value <= max_val):
            return False, f"第{i+1}个分量超出范围 [{min_val}, {max_val}]"

    return True, ""


def convert_multiple_colors(color_groups, target_space='RGB'):
    """
    转换多组颜色到目标颜色空间

    输入:
        color_groups: list - 颜色组列表，每个元素为 {'space': str, 'values': list}
        target_space: str - 目标颜色空间

    输出:
        list - 转换后的颜色组列表
    """
    converted_groups = []

    for group in color_groups:
        if group['space'] == target_space:
            converted_groups.append(group)
        else:
            # 转换颜色空间
            converted_color = convert_color_space(group['values'], group['space'], target_space)
            converted_groups.append({
                'space': target_space,
                'values': list(converted_color)
            })

    return converted_groups


def validate_multiple_colors(color_groups):
    """
    验证多组颜色值的有效性

    输入:
        color_groups: list - 颜色组列表

    输出:
        tuple - (是否全部有效, 错误信息列表)
    """
    all_valid = True
    error_messages = []

    for i, group in enumerate(color_groups):
        is_valid, error_msg = validate_color_values(group['values'], group['space'])
        if not is_valid:
            all_valid = False
            error_messages.append(f"第{i+1}组颜色: {error_msg}")

    return all_valid, error_messages