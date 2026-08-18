"""
Helper functions
"""

import os
import random
import sys
import numpy as np
from datetime import datetime


def get_default_dir():
    """
    Default directory for file dialogs: the project root (where main.py lives).

    Open/save dialogs default to the project directory regardless of
    where the app is launched, avoiding exposure of the user's private paths.
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_image_dir():
    """
    Default directory for image selection dialogs: the demo_images folder under the project root.

    Dedicated to demo/input images; created automatically if missing.
    """
    image_dir = os.path.join(get_default_dir(), 'demo_images')
    os.makedirs(image_dir, exist_ok=True)
    return image_dir


def safe_flush_stdout():
    """
    Safely flush standard output.

    In console-less environments (e.g. pythonw or detached processes),
    sys.stdout may be None or its underlying handle invalid,
    so calling flush() directly can raise OSError/ValueError.
    """
    try:
        if sys.stdout is not None:
            sys.stdout.flush()
    except (OSError, ValueError, AttributeError):
        pass


def normalize_color_values(values, color_space):
    """
    Normalize color values to their standard ranges.

    Args:
        values: list[float] - color values
        color_space: str - color space name

    Returns:
        normalized_values: list[float] - normalized color values
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
    Compute the distance between two colors.

    Args:
        color1: tuple/list - first color
        color2: tuple/list - second color
        color_space: str - color space

    Returns:
        distance: float - color distance
    """
    if len(color1) != 3 or len(color2) != 3:
        return float('inf')

    # Euclidean distance
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
    Generate a unique filename.

    Args:
        base_name: str - base file name
        extension: str - file extension

    Returns:
        filename: str - unique filename
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_id = random.randint(1000, 9999)

    # Strip illegal characters from the base name
    safe_base = ''.join(c for c in base_name if c.isalnum() or c in ('_', '-')).strip()
    if not safe_base:
        safe_base = 'design'

    return f"{safe_base}_{timestamp}_{random_id}.{extension.lstrip('.')}"


def rgb_to_hex(rgb):
    """
    Convert an RGB color to a hex string.

    Args:
        rgb: tuple - RGB color values

    Returns:
        str - hex color code
    """
    if len(rgb) >= 3:
        return '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0]), int(rgb[1]), int(rgb[2])
        )
    return '#000000'


def hex_to_rgb(hex_color):
    """
    Convert a hex color to RGB.

    Args:
        hex_color: str - hex color code

    Returns:
        tuple - RGB color values
    """
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 6:
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (0, 0, 0)
