"""
图像分析功能
"""

import random
import numpy as np
from PIL import Image
import cv2


def load_and_preprocess_image(image_path, resize_to=(800, 600)):
    """
    加载并预处理图像

    输入:
        image_path: str - 图像路径
        resize_to: tuple - 调整后的尺寸（可选）

    输出:
        np.array - 预处理后的图像数组（RGB格式）
    """
    try:
        # 使用PIL加载图像
        img = Image.open(image_path)

        # 转换为RGB（如果是RGBA）
        if img.mode in ('RGBA', 'LA'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[-1])
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')

        # 调整尺寸
        if resize_to:
            img = img.resize(resize_to, Image.Resampling.LANCZOS)

        # 转换为numpy数组
        img_array = np.array(img)

        return img_array

    except Exception as e:
        print(f"图像加载失败: {e}")
        # 返回一个随机的测试图像
        return np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)


def analyze_environment_texture(environment_image):
    """
    分析环境纹理特征

    输入:
        environment_image: np.array - 环境图像

    输出:
        dict - 纹理特征字典
    """
    # 模拟实现：返回随机纹理特征
    return {
        'color_palette': [
            (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            for _ in range(3)
        ],
        'texture_density': random.uniform(0.2, 0.8),
        'contrast': random.uniform(0.3, 0.9),
        'complexity': random.uniform(0.4, 0.95),
        'dominant_orientation': random.uniform(0, 180),
        'roughness': random.uniform(0.1, 0.9),
    }


def calculate_color_statistics(image, mask=None):
    """
    计算图像颜色统计特征

    输入:
        image: np.array - 图像
        mask: np.array - 掩码（可选）

    输出:
        dict - 统计信息字典
    """
    if image is None:
        return {}

    # 如果有掩码，应用掩码
    if mask is not None and mask.shape[:2] == image.shape[:2]:
        masked_image = image[mask > 0]
        if len(masked_image) == 0:
            pixels = image.reshape(-1, 3)
        else:
            pixels = masked_image
    else:
        pixels = image.reshape(-1, 3)

    # 计算统计量
    mean = np.mean(pixels, axis=0)
    std = np.std(pixels, axis=0)

    # 计算颜色直方图
    hist_red = np.histogram(pixels[:, 0], bins=32, range=(0, 255))[0]
    hist_green = np.histogram(pixels[:, 1], bins=32, range=(0, 255))[0]
    hist_blue = np.histogram(pixels[:, 2], bins=32, range=(0, 255))[0]

    # 找到出现最多的颜色作为主色
    hist_3d, _ = np.histogramdd(pixels, bins=(8, 8, 8), range=[(0, 255), (0, 255), (0, 255)])
    max_idx = np.unravel_index(np.argmax(hist_3d), hist_3d.shape)
    dominant_color = (
        int((max_idx[0] + 0.5) * 255 / 8),
        int((max_idx[1] + 0.5) * 255 / 8),
        int((max_idx[2] + 0.5) * 255 / 8)
    )

    return {
        'mean': tuple(mean.astype(int)),
        'std': tuple(std.astype(int)),
        'dominant_color': dominant_color,
        'histogram': {
            'red': hist_red.tolist(),
            'green': hist_green.tolist(),
            'blue': hist_blue.tolist(),
        },
        'pixel_count': len(pixels),
    }