"""
相似度度量函数
"""

import numpy as np
from skimage.metrics import structural_similarity as ssim_skimage


def ssim_rgb(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    计算RGB图像的SSIM（结构相似性）

    Args:
        img1, img2: RGB图像 (H×W×3)

    Returns:
        SSIM平均值
    """
    if img1.shape != img2.shape or len(img1.shape) != 3:
        raise ValueError("输入必须为相同尺寸的RGB图像 (H×W×3)")

    ssim_vals = np.zeros(3)
    for ch in range(3):
        # 计算数据范围
        data_range = np.max([img1[:, :, ch].max(), img2[:, :, ch].max()]) - \
                     np.min([img1[:, :, ch].min(), img2[:, :, ch].min()])

        ssim_vals[ch] = ssim_skimage(img1[:, :, ch], img2[:, :, ch],
                                     data_range=data_range)

    return np.mean(ssim_vals)


def uiqi(img1: np.ndarray, img2: np.ndarray, C1: float = 0.01, C2: float = 0.03) -> float:
    """
    计算通用图像质量指数 (UIQI)

    UIQI = (4 * σ_xy * μ_x * μ_y) / ((σ_x² + σ_y²) * (μ_x² + μ_y² + C1) * (σ_x * σ_y + C2))

    Args:
        img1, img2: 输入图像
        C1, C2: 稳定性常数

    Returns:
        UIQI值
    """
    # 确保图像尺寸相同
    if img1.shape != img2.shape:
        raise ValueError("图像尺寸必须相同")

    # 转换为浮点型
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    # 如果是RGB图像，计算每个通道的平均值
    if len(img1.shape) == 3:
        uiqi_vals = np.zeros(3)
        for ch in range(3):
            uiqi_vals[ch] = uiqi_single_channel(img1[:, :, ch], img2[:, :, ch], C1, C2)
        return np.mean(uiqi_vals)
    else:
        return uiqi_single_channel(img1, img2, C1, C2)


def uiqi_single_channel(x: np.ndarray, y: np.ndarray, C1: float = 0.01, C2: float = 0.03) -> float:
    """单通道UIQI计算"""
    # 计算均值
    μ_x = np.mean(x)
    μ_y = np.mean(y)

    # 计算标准差
    σ_x = np.std(x)
    σ_y = np.std(y)

    # 计算协方差
    σ_xy = np.mean((x - μ_x) * (y - μ_y))

    # 避免除以零
    if σ_x == 0 or σ_y == 0:
        return 0.0

    # 计算UIQI
    numerator = 4 * σ_xy * μ_x * μ_y
    denominator = (σ_x ** 2 + σ_y ** 2) * (μ_x ** 2 + μ_y ** 2 + C1) * (σ_x * σ_y + C2)

    return numerator / denominator


def calculate_stability(similarities: np.ndarray) -> float:
    """
    计算稳定性指标

    Args:
        similarities: 相似度分数数组

    Returns:
        稳定性分数 (0-1)
    """
    if len(similarities) < 2:
        return 0.0

    mean_sim = np.mean(similarities)
    std_sim = np.std(similarities)

    if mean_sim == 0:
        return 0.0

    # 稳定性 = 1 - 变异系数
    cv = std_sim / mean_sim
    stability = max(0.0, 1.0 - cv)

    return stability