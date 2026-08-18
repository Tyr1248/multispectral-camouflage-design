"""
Color processing utilities - modified to support a variable number of output colors
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
    Extract dominant colors and their weights from an image

    Args:
        image_path: str - path to the image file
        n_colors: int - number of colors to extract (None lets the clustering algorithm decide)
        color_space: str - color space ('RGB', 'Lab', 'XYZ')

    Returns:
        tuple: (colors, weights)
            colors: list of RGB tuples [(R,G,B), ...] - extracted colors
            weights: list of float - weight of each color (area fraction or representativeness)
    """
    from Cluster_extraction.extract_color_features import ColorFeatureExtractor
    extractor = ColorFeatureExtractor(
        n_colors=4,  # extract 4 colors
        show_progress=True  # show progress information
    )
    result = extractor.extract_colors(image_path)[0]  # single-image result, formatted as [colors, weights]
    colors = result[0]
    weights = result[1]
    # print(colors)
    # print(weights)
    # colors=[(113, 142, 74), (76, 105, 45), (145, 177, 94), (220, 222, 188)]
    # weights=[0.5700216293334961, 0.29250526428222656, 0.11685466766357422, 0.020618438720703125]
    return colors, weights


def convert_color_space(color, from_space, to_space):
    """
    Convert between color spaces

    Args:
        color: tuple/list - color values
        from_space: str - source color space
        to_space: str - target color space

    Returns:
        tuple - converted color values
    """
    if from_space == to_space:
        return tuple(color)

    if not COLOUR_SCIENCE_AVAILABLE:
        # Fall back to the simplified conversion if colour-science is unavailable
        return _simplified_color_conversion(color, from_space, to_space)

    # Convert using the colour-science library
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
            # Other conversions fall back to returning the original values
            return tuple(color)
    except Exception:
        return _simplified_color_conversion(color, from_space, to_space)


def _simplified_color_conversion(color, from_space, to_space):
    """Simplified color space conversion"""
    if from_space == 'RGB' and to_space == 'Lab':
        # Simplified RGB to Lab conversion
        r, g, b = color
        x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
        y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
        z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b

        # Simplified XYZ to Lab conversion
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
        # Simplified Lab to RGB conversion
        L, a, b = color

        # Simplified Lab to XYZ conversion
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

        # Simplified XYZ to RGB conversion
        r = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z
        g = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z
        b = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z

        # Clamp to the 0-255 range
        r = max(0, min(255, int(r)))
        g = max(0, min(255, int(g)))
        b = max(0, min(255, int(b)))

        return (r, g, b)

    else:
        # Other conversions fall back to returning the original values
        return tuple(color)


def validate_color_values(values, color_space):
    """
    Validate that color values are within the valid range

    Args:
        values: list - color values
        color_space: str - color space

    Returns:
        tuple - (is_valid, error_message)
    """
    if len(values) != 3:
        return False, "必须输入3个颜色分量"

    # Define the valid ranges for each color space
    ranges = {
        'RGB': [(0, 255), (0, 255), (0, 255)],
        'Lab': [(0, 100), (-128, 127), (-128, 127)],
        'XYZ': [(0, 0.95), (0, 1.0), (0, 1.09)],
    }

    if color_space not in ranges:
        return False, f"不支持的颜色空间: {color_space}"

    # Validate each component
    for i, (value, (min_val, max_val)) in enumerate(zip(values, ranges[color_space])):
        if not (min_val <= value <= max_val):
            return False, f"第{i+1}个分量超出范围 [{min_val}, {max_val}]"

    return True, ""


def convert_multiple_colors(color_groups, target_space='RGB'):
    """
    Convert multiple color groups to the target color space

    Args:
        color_groups: list - color groups, each element is {'space': str, 'values': list}
        target_space: str - target color space

    Returns:
        list - converted color groups
    """
    converted_groups = []

    for group in color_groups:
        if group['space'] == target_space:
            converted_groups.append(group)
        else:
            # Convert the color space
            converted_color = convert_color_space(group['values'], group['space'], target_space)
            converted_groups.append({
                'space': target_space,
                'values': list(converted_color)
            })

    return converted_groups


def validate_multiple_colors(color_groups):
    """
    Validate multiple color groups

    Args:
        color_groups: list - color groups

    Returns:
        tuple - (all_valid, list of error messages)
    """
    all_valid = True
    error_messages = []

    for i, group in enumerate(color_groups):
        is_valid, error_msg = validate_color_values(group['values'], group['space'])
        if not is_valid:
            all_valid = False
            error_messages.append(f"第{i+1}组颜色: {error_msg}")

    return all_valid, error_messages
