import os
import numpy as np
import cv2
from datetime import datetime
import json
import warnings
from typing import List, Tuple, Dict, Any, Optional

warnings.filterwarnings('ignore')

try:
    from Camo.digital_camouflage import DigitalCamouflageRandom
except ImportError as e:
    print(f"导入模块失败：{e}")
    exit(1)


def convert_to_serializable(obj):
    """
    Convert an object to a JSON-serializable format.

    Args:
        obj: Object of any type.
    Returns:
        JSON-serializable object.
    """
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


def replace_colors_in_pattern(pattern: np.ndarray,
                              source_colors: List[List[int]],
                              target_colors: List[List[int]]) -> np.ndarray:
    """
    Replace colors in a pattern from source colors to target colors.

    Args:
        pattern: Input pattern, shape (H, W, 3), dtype=np.uint8.
        source_colors: Source color list, length=n_colors, each color is [R, G, B].
        target_colors: Target color list, length=n_colors, each color is [R, G, B].
    Returns:
        result: Pattern after replacement, shape (H, W, 3), dtype=np.uint8.
    """
    if len(source_colors) != len(target_colors):
        raise ValueError(f"颜色列表长度不匹配：source={len(source_colors)}, target={len(target_colors)}")

    result = pattern.copy()
    height, width = pattern.shape[:2]

    source_array = np.array(source_colors, dtype=np.uint8).reshape(1, 1, -1, 3)
    target_array = np.array(target_colors, dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            pixel = pattern[y, x, :]
            distances = np.sqrt(np.sum((source_array[0, 0, :, :] - pixel) ** 2, axis=1))
            closest_idx = np.argmin(distances)

            if distances[closest_idx] < 10:
                result[y, x, :] = target_array[closest_idx]

    return result


def generate_camouflage_pattern(
        amorphous_colors: List[List[int]],
        crystalline_colors: List[List[int]],
        color_budget: List[float]
) -> Dict[str, Any]:
    """
    Generate camouflage patterns (uses a fixed color budget; no environment image).

    Args:
        amorphous_colors: Amorphous-state color list, length=n_colors, each color is [R, G, B].
        crystalline_colors: Crystalline-state color list, length=n_colors, each color is [R, G, B].
        color_budget: Color budget list, length=n_colors, usage ratio per color (auto-normalized).

    Returns:
        result: Dictionary with the following keys:
            - 'amorphous_pattern': np.ndarray, shape (1024, 1024, 3), dtype=np.uint8
            - 'crystalline_pattern': np.ndarray, shape (1024, 1024, 3), dtype=np.uint8
            - 'pattern_size': Tuple[int, int], (1024, 1024)
            - 'color_count': int, number of colors
            - 'environment_used': bool, always False
            - 'color_budget': List[float], normalized color budget
    """
    # Input validation
    if len(amorphous_colors) != len(crystalline_colors):
        raise ValueError(
            f"颜色列表长度不匹配：amorphous={len(amorphous_colors)}, crystalline={len(crystalline_colors)}")
    if len(amorphous_colors) != len(color_budget):
        raise ValueError(
            f"颜色数量与预算长度不匹配：colors={len(amorphous_colors)}, budget={len(color_budget)}")

    color_count = len(amorphous_colors)

    # Convert colors to numpy arrays
    amorphous_colors_rgb = np.array(amorphous_colors, dtype=np.uint8)
    crystalline_colors_rgb = np.array(crystalline_colors, dtype=np.uint8)

    # Normalize the color budget
    budget = np.array(color_budget, dtype=float)
    budget = budget / np.sum(budget)
    print(f"使用固定颜色预算：{budget.tolist()}")

    # ========== Size parameters ==========
    base_size = 256  # initial canvas: 256x256
    upscale_factor = 4  # upscaling factor: 4x
    final_size = base_size * upscale_factor  # 1024×1024
    base_seed = 2026

    final_patterns = []

    for i in range(3):
        try:
            seed = base_seed + i * 1000
            expand_pixels = min(10, base_size // 10)

            generator = DigitalCamouflageRandom(
                canvas_size=(base_size, base_size),
                spot_database_path='spot_database',
                expand_pixels=expand_pixels,
                upscale_factor=upscale_factor
            )

            camouflage_upscaled = generator.generate_camouflage_pattern(
                amorphous_colors_rgb, 1.0, budget.tolist(), seed=seed
            )

            final_patterns.append(camouflage_upscaled)
        except Exception as e:
            print(f"生成失败：{e}")
            continue

    if not final_patterns:
        # Fallback: generate vertical stripes
        amorphous_pattern = np.zeros((final_size, final_size, 3), dtype=np.uint8)
        for i in range(color_count):
            start_x = (final_size // color_count) * i
            end_x = (final_size // color_count) * (i + 1)
            amorphous_pattern[:, start_x:end_x, :] = amorphous_colors_rgb[i]
    else:
        amorphous_pattern = final_patterns[0]

    # Generate the crystalline-state camouflage pattern
    crystalline_pattern = replace_colors_in_pattern(
        amorphous_pattern, amorphous_colors, crystalline_colors
    )

    # Prepare the result
    result = {
        'amorphous_pattern': amorphous_pattern,
        'crystalline_pattern': crystalline_pattern,
        'pattern_size': (final_size, final_size),
        'color_count': color_count,
        'environment_used': False,
        'color_budget': budget.tolist()
    }

    return result


def main():
    """Example usage."""
    amorphous_colors =  [
    [
      167,
      151,
      142
    ],
    [
      139,
      127,
      112
    ],
    [
      178,
      167,
      149
    ],
    [
      84,
      75,
      57
    ]
  ]

    crystalline_colors =  [
    [
      180,
      183,
      145
    ],
    [
      110,
      135,
      103
    ],
    [
      164,
      192,
      144
    ],
    [
      90,
      116,
      70
    ]
  ]

    fixed_budget =  [
    0.3969, 0.3583, 0.1936, 0.0512
  ]

    result = generate_camouflage_pattern(
        amorphous_colors=amorphous_colors,
        crystalline_colors=crystalline_colors,
        color_budget=fixed_budget
    )

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    result_dir = f'camouflage_results_{timestamp}'
    os.makedirs(result_dir, exist_ok=True)

    amorphous_bgr = cv2.cvtColor(result['amorphous_pattern'], cv2.COLOR_RGB2BGR)
    crystalline_bgr = cv2.cvtColor(result['crystalline_pattern'], cv2.COLOR_RGB2BGR)

    cv2.imwrite(os.path.join(result_dir, 'amorphous_pattern.png'), amorphous_bgr)
    cv2.imwrite(os.path.join(result_dir, 'crystalline_pattern.png'), crystalline_bgr)

    metadata = {
        'pattern_size': result['pattern_size'],
        'color_count': result['color_count'],
        'environment_used': result['environment_used'],
        'color_budget': result['color_budget'],
        'amorphous_colors': amorphous_colors,
        'crystalline_colors': crystalline_colors
    }

    with open(os.path.join(result_dir, 'metadata.json'), 'w') as f:
        json.dump(convert_to_serializable(metadata), f, indent=2)

    print(f"\n结果已保存至：{result_dir}")
    print(f"  输出尺寸：{result['pattern_size']}")


if __name__ == "__main__":
    main()