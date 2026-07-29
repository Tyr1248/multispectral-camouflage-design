import numpy as np
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')


@dataclass
class OptimizationResult:
    """优化结果"""
    best_budget: np.ndarray
    best_score: float


class GradientOptimizer:
    """梯度搜索优化器（简单高效版）"""

    def __init__(self, n_seeds: int = 3):
        self.n_seeds = n_seeds

    def optimize_budget(self,
                        dominant_colors_rgb: np.ndarray,
                        color_analysis_result,
                        background_blocks: List[np.ndarray]) -> OptimizationResult:
        """
        简化优化：基于规则生成预算

        规则：
        1. 仅前两种颜色能被选作主色，且必定包含第一个颜色
        2. 主色为2时，主色预算范围0.3-0.7，副色预算大于0.07
        """
        n_colors = dominant_colors_rgb.shape[0]
        n_dominant = color_analysis_result.n_dominant_colors
        dominant_indices = color_analysis_result.dominant_color_indices

        # 如果背景块不足，使用默认预算
        if len(background_blocks) == 0:
            return self.get_default_budget(n_colors)

        # 1. 计算颜色与背景的相似度
        color_similarities = self.calculate_color_similarities(
            dominant_colors_rgb, background_blocks
        )

        # 2. 根据规则分配预算
        if n_dominant == 1:
            best_budget = self.optimize_single_dominant(
                n_colors, color_similarities
            )
        else:  # n_dominant == 2
            best_budget = self.optimize_double_dominant(
                n_colors, color_similarities
            )

        # 3. 评估预算得分
        best_score = self.evaluate_budget(
            best_budget, dominant_colors_rgb, background_blocks
        )

        return OptimizationResult(
            best_budget=best_budget,
            best_score=best_score
        )

    def calculate_color_similarities(self, colors: np.ndarray,
                                     background_blocks: List[np.ndarray]) -> np.ndarray:
        """计算颜色与背景的相似度"""
        from Camo.similarity_metrics import ssim_rgb

        n_colors = colors.shape[0]
        color_similarities = np.zeros(n_colors)

        for i in range(min(n_colors, 2)):  # 只计算前两个颜色
            similarities = []
            for block in background_blocks:
                # 创建单色测试图像
                test_block = np.full_like(block, colors[i], dtype=np.uint8)
                similarity = ssim_rgb(test_block, block)
                similarities.append(similarity)
            color_similarities[i] = np.mean(similarities)

        return color_similarities

    def optimize_single_dominant(self, n_colors: int,
                                 color_similarities: np.ndarray) -> np.ndarray:
        """优化单主色情况"""
        budget = np.zeros(n_colors)

        # 第一个颜色为主色
        budget[0] = 0.7  # 固定70%

        # 分配剩余预算
        if n_colors > 1:
            remaining = 0.3
            # 确保副色预算大于0.07
            min_per_color = 0.07
            total_min = min_per_color * (n_colors - 1)

            if remaining >= total_min:
                # 平均分配
                for i in range(1, n_colors):
                    budget[i] = remaining / (n_colors - 1)
            else:
                # 需要从主色调整
                need = total_min - remaining
                budget[0] -= need  # 从主色扣除
                for i in range(1, n_colors):
                    budget[i] = min_per_color

        # 归一化
        budget = budget / np.sum(budget)
        return budget

    def optimize_double_dominant(self, n_colors: int,
                                 color_similarities: np.ndarray) -> np.ndarray:
        """优化双主色情况"""
        budget = np.zeros(n_colors)

        # 根据相似度分配主色预算
        sim0 = color_similarities[0]
        sim1 = color_similarities[1] if n_colors > 1 else 0

        # 确保有足够的相似度信息
        if sim0 + sim1 == 0:
            # 默认分配
            budget[0] = 0.5
            if n_colors > 1:
                budget[1] = 0.3
        else:
            # 按相似度比例分配，但保持在0.3-0.7范围内
            base_main = 0.8  # 两个主色共占80%

            # 第一个颜色最小0.3，最大0.7
            main1 = 0.3 + 0.4 * (sim0 / (sim0 + sim1))
            main1 = np.clip(main1, 0.3, 0.7)

            # 第二个颜色同样范围
            main2 = base_main - main1
            main2 = np.clip(main2, 0.3, 0.7)

            # 调整第一个颜色以确保和为base_main
            main1 = base_main - main2

            budget[0] = main1
            if n_colors > 1:
                budget[1] = main2

        # 分配副色预算
        if n_colors > 2:
            used = budget[0] + budget[1]
            remaining = 1.0 - used
            other_colors = n_colors - 2

            # 确保副色预算大于0.07
            min_per_color = 0.07
            total_min = min_per_color * other_colors

            if remaining >= total_min:
                # 平均分配剩余预算
                for i in range(2, n_colors):
                    budget[i] = remaining / other_colors
            else:
                # 需要从主色调整
                need = total_min - remaining
                # 从两个主色中均摊扣除
                adjust_per_main = need / 2
                budget[0] -= adjust_per_main
                budget[1] -= adjust_per_main

                for i in range(2, n_colors):
                    budget[i] = min_per_color

        # 确保所有值都大于0
        budget = np.maximum(budget, 0.01)
        # 归一化
        budget = budget / np.sum(budget)

        return budget

    def get_default_budget(self, n_colors: int) -> OptimizationResult:
        """获取默认预算（无背景图像时使用）"""
        if n_colors == 1:
            budget = np.array([1.0])
        elif n_colors == 2:
            budget = np.array([0.7, 0.3])
        elif n_colors == 3:
            budget = np.array([0.7, 0.15, 0.15])
        else:  # n_colors >= 4
            budget = np.array([0.7, 0.1, 0.1, 0.1])
            if n_colors > 4:
                remaining = 0.1
                additional = np.ones(n_colors - 4) * (remaining / (n_colors - 4))
                budget = np.concatenate([budget[:4], additional])

        budget = budget / np.sum(budget)

        return OptimizationResult(
            best_budget=budget,
            best_score=0.5  # 默认得分
        )

    def evaluate_budget(self, budget: np.ndarray,
                        colors: np.ndarray,
                        background_blocks: List[np.ndarray]) -> float:
        """评估预算得分"""
        import cv2
        from Camo.digital_camouflage import DigitalCamouflageRandom
        from Camo.similarity_metrics import ssim_rgb, uiqi

        scores = []

        for seed in range(min(self.n_seeds, 2)):  # 减少评估次数
            try:
                generator = DigitalCamouflageRandom(
                    canvas_size=(64, 64),
                    spot_database_path='spot_database',
                    expand_pixels=6
                )

                camouflage = generator.generate_camouflage_pattern(
                    colors, 1.0, budget.tolist(), seed=seed * 1000
                )

                # 上采样到256x256
                camouflage = cv2.resize(
                    camouflage, (256, 256), interpolation=cv2.INTER_NEAREST
                )

                # 计算相似度
                block_scores = []
                for block in background_blocks:
                    ssim_val = ssim_rgb(camouflage, block)
                    # uiqi_val = uiqi(camouflage, block)
                    combined = 0.6 * ssim_val
                    block_scores.append(combined)

                avg_score = np.mean(block_scores)
                scores.append(avg_score)
            except Exception:
                continue

        return np.mean(scores) if scores else 0.5