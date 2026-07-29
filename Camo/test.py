"""
测试迷彩生成函数
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

from camouflage_generator import generate_camouflage_pattern


def display_pattern(pattern: np.ndarray, title: str):
    """显示迷彩图案"""
    plt.figure(figsize=(8, 8))
    plt.imshow(pattern)
    plt.title(title)
    plt.axis('off')
    plt.show()


def save_pattern(pattern: np.ndarray, filename: str):
    """保存迷彩图案"""
    # 转换为BGR格式
    pattern_bgr = cv2.cvtColor(pattern, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, pattern_bgr)
    print(f"保存图案到: {filename}")


def test_without_environment():
    """测试不使用环境图像的情况"""
    print("测试1: 不使用环境图像")

    # 定义非晶态和晶态颜色
    amorphous_colors = [
        [163, 159, 136],  # 颜色1
        [95, 130, 61],  # 颜色2
        [81, 96, 63],  # 颜色3
        [123, 149, 69]  # 颜色4
    ]

    crystalline_colors = [
        [200, 180, 150],  # 对应的晶态颜色1
        [150, 180, 100],  # 对应的晶态颜色2
        [130, 140, 100],  # 对应的晶态颜色3
        [180, 200, 120]  # 对应的晶态颜色4
    ]

    # 生成迷彩图案
    result = generate_camouflage_pattern(amorphous_colors, crystalline_colors)

    print(f"颜色数量: {result['color_count']}")
    print(f"使用环境图像: {result['environment_used']}")
    print(f"图案尺寸: {result['pattern_size']}")

    # 显示图案
    display_pattern(result['amorphous_pattern'], "非晶态迷彩图案")
    display_pattern(result['crystalline_pattern'], "晶态迷彩图案")

    # 保存图案
    save_pattern(result['amorphous_pattern'], "amorphous_pattern.png")
    save_pattern(result['crystalline_pattern'], "crystalline_pattern.png")

    return result


def test_with_environment():
    """测试使用环境图像的情况"""
    print("\n测试2: 使用环境图像")

    # 定义非晶态和晶态颜色
    amorphous_colors = [
        [163, 159, 136],
        [95, 130, 61],
        [81, 96, 63],
        [123, 149, 69]
    ]

    crystalline_colors = [
        [200, 180, 150],
        [150, 180, 100],
        [130, 140, 100],
        [180, 200, 120]
    ]

    # 使用环境图像（假设有环境图像文件）
    environment_paths = [
        "ukraine_4096x2048_512x512.png",  # 示例环境图像
        # 可以添加更多环境图像
    ]

    # 生成迷彩图案
    result = generate_camouflage_pattern(
        amorphous_colors,
        crystalline_colors,
        environment_paths
    )

    print(f"颜色数量: {result['color_count']}")
    print(f"使用环境图像: {result['environment_used']}")
    print(f"颜色预算: {result.get('color_budget', 'N/A')}")

    # 显示图案
    display_pattern(result['amorphous_pattern'], "非晶态迷彩图案 (基于环境)")
    display_pattern(result['crystalline_pattern'], "晶态迷彩图案 (基于环境)")

    # 保存图案
    save_pattern(result['amorphous_pattern'], "amorphous_pattern_env.png")
    save_pattern(result['crystalline_pattern'], "crystalline_pattern_env.png")

    return result


def test_different_color_count():
    """测试不同颜色数量"""
    print("\n测试3: 不同颜色数量")

    # 3种颜色
    amorphous_colors_3 = [
        [163, 159, 136],
        [95, 130, 61],
        [81, 96, 63]
    ]

    crystalline_colors_3 = [
        [200, 180, 150],
        [150, 180, 100],
        [130, 140, 100]
    ]

    result_3 = generate_camouflage_pattern(amorphous_colors_3, crystalline_colors_3)
    print(f"3种颜色 - 图案尺寸: {result_3['pattern_size']}")

    # 5种颜色
    amorphous_colors_5 = [
        [163, 159, 136],
        [95, 130, 61],
        [81, 96, 63],
        [123, 149, 69],
        [140, 120, 80]
    ]

    crystalline_colors_5 = [
        [200, 180, 150],
        [150, 180, 100],
        [130, 140, 100],
        [180, 200, 120],
        [170, 150, 110]
    ]

    result_5 = generate_camouflage_pattern(amorphous_colors_5, crystalline_colors_5)
    print(f"5种颜色 - 图案尺寸: {result_5['pattern_size']}")

    return result_3, result_5


def main():
    """主测试函数"""
    print("迷彩图案生成器测试")
    print("=" * 60)

    # 测试1: 不使用环境图像
    test1_result = test_without_environment()

    # 测试2: 使用环境图像（如果有环境图像文件）
    try:
        test2_result = test_with_environment()
    except Exception as e:
        print(f"测试2失败: {e}")
        print("请确保有有效的环境图像文件")

    # 测试3: 不同颜色数量
    test3_results = test_different_color_count()

    print("\n所有测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()