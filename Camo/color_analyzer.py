import numpy as np
from typing import List, Tuple, Dict, Any
import cv2
from dataclasses import dataclass
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings('ignore')


@dataclass
class ColorAnalysisResult:
    """颜色分析结果"""
    n_dominant_colors: int
    dominant_color_indices: List[int]
    background_dominant_colors: np.ndarray
    input_colors: np.ndarray
    color_similarity_matrix: np.ndarray


class ColorAnalyzer:
    """颜色空间分析器"""

    def __init__(self, color_distance_threshold: float = 20.0):
        self.color_distance_threshold = color_distance_threshold

    def analyze_colors_from_background(self, background_img: np.ndarray,
                                       input_colors_rgb: np.ndarray) -> ColorAnalysisResult:
        """
        从背景图像分析颜色

        规则：仅前两种颜色能被选作主色，且必定包含第一个颜色

        Args:
            background_img: 背景图像 (HxWx3, RGB)
            input_colors_rgb: 输入颜色数组 (n_colors x 3, RGB)

        Returns:
            ColorAnalysisResult: 颜色分析结果
        """
        n_input_colors = input_colors_rgb.shape[0]

        # 1. 从背景图像提取主要颜色
        background_colors = self.extract_dominant_colors_from_background(
            background_img, n_colors=min(6, n_input_colors + 2)
        )

        # 2. 计算颜色相似度
        similarity_matrix = self.calculate_color_similarities(input_colors_rgb, background_colors)

        # 3. 根据规则确定主色
        # 规则1：仅前两种颜色能被选作主色
        # 规则2：必定包含第一个颜色

        if n_input_colors == 1:
            # 只有一个颜色
            n_dominant = 1
            dominant_indices = [0]
        else:
            # 计算前两个颜色与背景的相似度
            color_background_similarity = np.max(similarity_matrix, axis=1)

            # 第一个颜色总是主色
            # 决定第二个颜色是否为主色
            if n_input_colors >= 2:
                # 计算前两个颜色的差异
                input_colors_lab = self.rgb_to_lab(input_colors_rgb[:2])
                color_distance = np.sqrt(np.sum((input_colors_lab[0] - input_colors_lab[1]) ** 2))

                # 第二个颜色与背景的相似度
                sim_second = color_background_similarity[1]

                # 决定主色数量
                if color_distance > self.color_distance_threshold and sim_second > 0.5:
                    # 第二个颜色与第一个差异大，且与背景相似度高，作为第二个主色
                    n_dominant = 2
                    dominant_indices = [0, 1]
                else:
                    # 只使用一个主色
                    n_dominant = 1
                    dominant_indices = [0]
            else:
                n_dominant = 1
                dominant_indices = [0]

        return ColorAnalysisResult(
            n_dominant_colors=n_dominant,
            dominant_color_indices=dominant_indices,
            background_dominant_colors=background_colors,
            input_colors=input_colors_rgb,
            color_similarity_matrix=similarity_matrix
        )

    def extract_dominant_colors_from_background(self, background_img: np.ndarray,
                                                n_colors: int = 6) -> np.ndarray:
        """从背景图像提取主要颜色"""
        height, width = background_img.shape[:2]
        max_pixels = 10000
        if height * width > max_pixels:
            scale = np.sqrt(max_pixels / (height * width))
            new_height = int(height * scale)
            new_width = int(width * scale)
            resized_img = cv2.resize(background_img, (new_width, new_height))
        else:
            resized_img = background_img

        pixels = resized_img.reshape(-1, 3)

        if len(pixels) >= n_colors:
            kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
            kmeans.fit(pixels)
            dominant_colors = kmeans.cluster_centers_.astype(np.uint8)

            labels = kmeans.labels_
            cluster_sizes = np.bincount(labels)
            sorted_indices = np.argsort(cluster_sizes)[::-1]
            dominant_colors = dominant_colors[sorted_indices]
        else:
            indices = np.random.choice(len(pixels), min(n_colors, len(pixels)), replace=False)
            dominant_colors = pixels[indices].astype(np.uint8)

        return dominant_colors

    def calculate_color_similarities(self, colors1: np.ndarray, colors2: np.ndarray) -> np.ndarray:
        """计算颜色相似度矩阵"""
        n1 = colors1.shape[0]
        n2 = colors2.shape[0]
        similarity_matrix = np.zeros((n1, n2))

        colors1_lab = self.rgb_to_lab(colors1)
        colors2_lab = self.rgb_to_lab(colors2)

        for i in range(n1):
            for j in range(n2):
                delta_e = np.sqrt(np.sum((colors1_lab[i] - colors2_lab[j]) ** 2))
                similarity_matrix[i, j] = 1.0 / (1.0 + delta_e / 100.0)

        return similarity_matrix

    def rgb_to_lab(self, colors_rgb: np.ndarray) -> np.ndarray:
        """RGB转CIELAB颜色空间"""
        colors_rgb_uint8 = np.clip(colors_rgb, 0, 255).astype(np.uint8)
        n_colors = colors_rgb_uint8.shape[0]
        colors_lab = np.zeros_like(colors_rgb_uint8, dtype=np.float32)

        for i in range(n_colors):
            color_rgb = colors_rgb_uint8[i].reshape(1, 1, 3)
            color_lab = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2LAB)
            colors_lab[i] = color_lab[0, 0]

        return colors_lab