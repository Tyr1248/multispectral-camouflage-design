import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import warnings
from dataclasses import dataclass


@dataclass
class BudgetFeedback:
    """颜色预算反馈控制数据结构（支持可变颜色数量）"""
    target_ratios: np.ndarray
    current_usage: np.ndarray
    total_placed: int
    target_pixels: np.ndarray

    @property
    def usage_ratios(self) -> np.ndarray:
        """当前使用比例"""
        if self.total_placed == 0:
            return np.zeros_like(self.target_ratios)
        return self.current_usage / self.total_placed

    @property
    def deviation_ratios(self) -> np.ndarray:
        """与目标比例的偏差率"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return (self.usage_ratios - self.target_ratios) / (self.target_ratios + 1e-10)

    @property
    def remaining_budget_ratios(self) -> np.ndarray:
        """剩余预算占比（相对所有颜色剩余预算总和）"""
        remaining = self.get_remaining_budget()
        total_remaining = np.sum(remaining)
        if total_remaining < 1e-10:
            return np.zeros_like(remaining)
        return remaining / total_remaining

    def update_usage(self, color_idx: int, added_pixels: int):
        """更新颜色使用情况"""
        self.current_usage[color_idx] += added_pixels
        self.total_placed += added_pixels

    def get_remaining_budget(self) -> np.ndarray:
        """获取剩余预算（目标像素数 - 当前使用）"""
        return np.maximum(self.target_pixels - self.current_usage, 0)

    def get_remaining_budget_ratio(self, color_idx: int) -> float:
        """获取单个颜色的剩余预算占比"""
        return self.remaining_budget_ratios[color_idx]


class DigitalCamouflageRandom:
    """生成随机数字迷彩图案（4 倍上采样输出版本）"""

    def __init__(self, canvas_size: Tuple[int, int] = (256, 256),
                 spot_database_path: str = 'spot_database',
                 expand_pixels: int = 10,
                 upscale_factor: int = 4):
        """
        初始化迷彩生成器

        参数:
            canvas_size: 初始画布尺寸 (宽，高)，默认 (256, 256)
            spot_database_path: 斑点数据库路径
            expand_pixels: 边缘扩展像素数，默认 10
            upscale_factor: 上采样倍数，默认 4（输出尺寸 = canvas_size × upscale_factor）
        """
        self.original_canvas_size = canvas_size
        self.expanded_canvas_size = (canvas_size[0] + expand_pixels * 2,
                                     canvas_size[1] + expand_pixels * 2)
        self.expand_pixels = expand_pixels
        self.upscale_factor = upscale_factor

        # 上采样后的尺寸
        self.upscaled_canvas_size = (canvas_size[0] * upscale_factor,
                                     canvas_size[1] * upscale_factor)
        self.upscaled_expand_pixels = expand_pixels * upscale_factor

        # 初始化斑点数据库管理器
        from Camo.spot_database_manager import SpotDatabaseManager
        from Camo.spot_renderer import SpotRenderer
        self.spot_database_manager = SpotDatabaseManager(spot_database_path)
        self.spot_renderer = SpotRenderer()

        print(f"初始化迷彩生成器（{upscale_factor}倍上采样）: "
              f"原始尺寸={canvas_size}, 输出尺寸={self.upscaled_canvas_size}")

    def generate_camouflage_pattern(self, dominant_colors_rgb: np.ndarray,
                                    total_coverage_target: float,
                                    target_ratios: List[float],
                                    seed: int = None) -> np.ndarray:
        """
        生成迷彩图案（上采样输出版本）

        参数:
            dominant_colors_rgb: 主导颜色列表，形状 (n_colors, 3)，dtype=np.uint8
            total_coverage_target: 总覆盖率目标，0.0~1.0
            target_ratios: 各颜色目标比例列表，长度=n_colors，自动归一化
            seed: 随机种子，可选

        返回:
            canvas: 迷彩图案图像，形状 (H×factor, W×factor, 3)，dtype=np.uint8
                   其中 H,W 为 canvas_size，factor 为 upscale_factor
                   例如：canvas_size=(256,256), upscale_factor=4 → 输出 (1024, 1024, 3)
        """
        if seed is not None:
            np.random.seed(seed)

        # 验证和调整颜色预算
        target_ratios = self.validate_and_adjust_budgets(target_ratios)

        width, height = self.expanded_canvas_size
        n_colors = len(dominant_colors_rgb)

        # 创建空白画布（低分辨率）
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        coverage_mask = np.zeros((height, width), dtype=bool)

        # 初始化颜色预算反馈控制器
        total_pixels = width * height
        target_covered_pixels = int(total_coverage_target * total_pixels)

        budget_feedback = BudgetFeedback(
            target_ratios=np.array(target_ratios),
            current_usage=np.zeros(n_colors),
            total_placed=0,
            target_pixels=np.array(target_ratios) * target_covered_pixels
        )

        print(f"开始生成迷彩 - 颜色数量：{n_colors}, 目标颜色比例：{target_ratios}")
        print(f"  低分辨率画布：{width}×{height}")
        print(f"  上采样倍数：{self.upscale_factor}")
        print(f"  最终输出：{self.upscaled_canvas_size[0]}×{self.upscaled_canvas_size[1]}")

        # 阶段 1：网格化基础放置（应用预算 - 斑点大小绑定策略）
        print("\n阶段 1: 网格化基础放置（预算 - 斑点大小绑定）")
        canvas, coverage_mask = self.grid_based_placement(
            canvas, coverage_mask, dominant_colors_rgb, budget_feedback,
            coverage_target=0.7
        )

        # 阶段 2：间隙填充（全部使用小斑点）
        print("\n阶段 2: 间隙填充（全部小斑点）")
        canvas, coverage_mask = self.gap_filling(
            canvas, coverage_mask, dominant_colors_rgb, budget_feedback,
            coverage_target=0.9
        )

        # 阶段 3：像素级精修（确保 100% 覆盖）
        print("\n阶段 3: 像素级精修")
        canvas, coverage_mask = self.pixel_level_refinement(
            canvas, coverage_mask, dominant_colors_rgb, budget_feedback
        )

        # 验证覆盖率
        current_coverage = np.sum(coverage_mask) / total_pixels
        final_usage_ratios = budget_feedback.usage_ratios

        print(f"\n低分辨率生成完成:")
        print(f"  覆盖率：{current_coverage * 100:.1f}%")
        print(f"  颜色比例：{final_usage_ratios}")
        print(f"  最大偏差：{np.max(np.abs(final_usage_ratios - target_ratios)) * 100:.1f}%")

        # 裁切回原始尺寸（低分辨率）
        from Camo.image_processor import ImageProcessor
        canvas = ImageProcessor.crop_to_original_size(
            canvas, self.expand_pixels, self.original_canvas_size
        )

        # 上采样到最终尺寸
        print(f"\n阶段 4: 上采样 {self.upscale_factor}x")
        canvas = self.upscale_image(canvas)
        print(f"  输出尺寸：{canvas.shape[1]}×{canvas.shape[0]}")

        return canvas

    def upscale_image(self, canvas: np.ndarray) -> np.ndarray:
        """
        使用最近邻插值上采样图像
        保持色块边缘清晰，不产生模糊

        参数:
            canvas: 输入图像，形状 (H, W, 3)，dtype=np.uint8
        返回:
            upscaled: 上采样后图像，形状 (H×factor, W×factor, 3)，dtype=np.uint8
        """
        h, w, c = canvas.shape
        factor = self.upscale_factor

        # 使用 np.repeat 进行最近邻上采样
        upscaled = np.repeat(canvas, factor, axis=0)  # 行方向×factor
        upscaled = np.repeat(upscaled, factor, axis=1)  # 列方向×factor

        return upscaled

    def select_spot_type_by_budget(self, budget_feedback: BudgetFeedback,
                                   color_idx: int) -> str:
        """
        根据颜色预算占比选择斑点类型

        预算占比规则：
        - > 0.6: 100% 大斑点
        - 0.3 ~ 0.6: 大小斑点混合（大 40% : 小 60%）
        - ≤ 0.3: 100% 小斑点

        参数:
            budget_feedback: 预算反馈对象
            color_idx: 颜色索引
        返回:
            spot_type: 'large' 或 'small'
        """
        budget_ratio = budget_feedback.get_remaining_budget_ratio(color_idx)

        if budget_ratio > 0.6:
            return 'large'
        elif budget_ratio > 0.3:
            return 'large' if np.random.random() < 0.4 else 'small'
        else:
            return 'small'

    def grid_based_placement(self, canvas, coverage_mask, colors,
                             budget_feedback: BudgetFeedback, coverage_target: float):
        """
        网格化基础放置：在规则网格上放置斑点确保基本覆盖
        应用预算 - 斑点大小绑定策略

        参数:
            canvas: 画布数组，形状 (H, W, 3)
            coverage_mask: 覆盖掩码，形状 (H, W)
            colors: 颜色数组，形状 (n_colors, 3)
            budget_feedback: 预算反馈对象
            coverage_target: 目标覆盖率
        返回:
            canvas, coverage_mask: 更新后的画布和掩码
        """
        height, width, _ = canvas.shape
        total_pixels = height * width
        target_pixels = int(total_pixels * coverage_target)

        # 计算网格大小
        avg_spot_size = 12
        grid_size = max(avg_spot_size // 2, 8)

        # 创建网格点
        grid_x = np.arange(grid_size, width - grid_size, grid_size)
        grid_y = np.arange(grid_size, height - grid_size, grid_size)

        # 打乱网格点顺序
        grid_positions = []
        for x in grid_x:
            for y in grid_y:
                grid_positions.append((x, y))

        np.random.shuffle(grid_positions)

        placed_pixels = 0
        max_attempts = len(grid_positions) * 10
        spot_count = 0

        for attempt in range(min(max_attempts, len(grid_positions))):
            if placed_pixels >= target_pixels:
                break

            x, y = grid_positions[attempt]

            # 根据预算选择颜色
            color_idx = self.select_color_by_budget(budget_feedback)
            color = colors[color_idx]

            # 根据颜色预算选择斑点大小
            spot_type = self.select_spot_type_by_budget(budget_feedback, color_idx)

            # 获取斑点
            spot_cell = self.spot_database_manager.spot_database.get(spot_type, [])
            if not spot_cell:
                continue

            spot_data = spot_cell[np.random.randint(len(spot_cell))]
            rotation_angle = np.random.choice([0, 90, 180, 270])
            rotated_spot_data = self.spot_renderer.rotate_spot(spot_data, rotation_angle)
            spot_img = rotated_spot_data['image']
            sh, sw = spot_img.shape

            # 确保斑点不会超出边界
            if x + sw >= width or y + sh >= height:
                continue

            # 放置斑点
            added = 0
            for i in range(sh):
                for j in range(sw):
                    if spot_img[i, j]:
                        if not coverage_mask[y + i, x + j]:
                            canvas[y + i, x + j, :] = color
                            coverage_mask[y + i, x + j] = True
                            added += 1

            if added > 0:
                budget_feedback.update_usage(color_idx, added)
                placed_pixels += added
                spot_count += 1

        print(f"网格化放置完成：{placed_pixels}/{target_pixels} 像素")
        print(f"  斑点数量：{spot_count}")
        return canvas, coverage_mask

    def gap_filling(self, canvas, coverage_mask, colors,
                    budget_feedback: BudgetFeedback, coverage_target: float):
        """
        间隙填充：在空白区域放置小斑点
        全部使用小斑点，不考虑预算

        参数:
            canvas: 画布数组，形状 (H, W, 3)
            coverage_mask: 覆盖掩码，形状 (H, W)
            colors: 颜色数组，形状 (n_colors, 3)
            budget_feedback: 预算反馈对象
            coverage_target: 目标覆盖率
        返回:
            canvas, coverage_mask: 更新后的画布和掩码
        """
        height, width, _ = canvas.shape
        total_pixels = height * width
        target_pixels = int(total_pixels * coverage_target)

        placed_pixels = np.sum(coverage_mask)
        attempts = 0
        max_attempts = 20000
        spot_count = 0

        while placed_pixels < target_pixels and attempts < max_attempts:
            uncovered_y, uncovered_x = np.where(~coverage_mask)
            if len(uncovered_y) == 0:
                break

            idx = np.random.randint(len(uncovered_y))
            x, y = uncovered_x[idx], uncovered_y[idx]

            # 固定使用小斑点
            spot_type = 'small'
            color_idx = self.select_color_by_budget(budget_feedback)
            color = colors[color_idx]

            spot_cell = self.spot_database_manager.spot_database.get(spot_type, [])
            if not spot_cell:
                attempts += 1
                continue

            spot_data = spot_cell[np.random.randint(len(spot_cell))]
            rotation_angle = np.random.choice([0, 90, 180, 270])
            rotated_spot_data = self.spot_renderer.rotate_spot(spot_data, rotation_angle)
            spot_img = rotated_spot_data['image']
            sh, sw = spot_img.shape

            x = max(0, min(x, width - sw))
            y = max(0, min(y, height - sh))

            overlap_ratio = self.spot_renderer.calculate_overlap_ratio(
                coverage_mask, x, y, sw, sh, spot_img)

            if overlap_ratio > 0.6:
                attempts += 1
                continue

            added = 0
            for i in range(sh):
                for j in range(sw):
                    if spot_img[i, j] and not coverage_mask[y + i, x + j]:
                        canvas[y + i, x + j, :] = color
                        coverage_mask[y + i, x + j] = True
                        added += 1

            if added > 0:
                budget_feedback.update_usage(color_idx, added)
                placed_pixels += added
                spot_count += 1

            attempts += 1

        print(f"间隙填充完成：{placed_pixels}/{target_pixels} 像素")
        print(f"  斑点数量：{spot_count}")
        return canvas, coverage_mask

    def pixel_level_refinement(self, canvas, coverage_mask, colors,
                               budget_feedback: BudgetFeedback):
        """
        像素级精修：确保 100% 覆盖并满足颜色预算

        参数:
            canvas: 画布数组，形状 (H, W, 3)
            coverage_mask: 覆盖掩码，形状 (H, W)
            colors: 颜色数组，形状 (n_colors, 3)
            budget_feedback: 预算反馈对象
        返回:
            canvas, coverage_mask: 更新后的画布和掩码
        """
        height, width, _ = canvas.shape

        uncovered_indices = np.where(~coverage_mask)
        uncovered_y, uncovered_x = uncovered_indices
        num_uncovered = len(uncovered_y)

        if num_uncovered == 0:
            print("像素级精修：已完全覆盖")
            return canvas, coverage_mask

        print(f"像素级精修：需要填充 {num_uncovered} 个像素")

        indices = np.random.permutation(num_uncovered)

        remaining_budget = budget_feedback.get_remaining_budget()
        total_remaining = np.sum(remaining_budget)

        if total_remaining <= 0:
            probs = budget_feedback.current_usage / np.sum(budget_feedback.current_usage)
        else:
            probs = remaining_budget / total_remaining

        probs = probs / np.sum(probs)

        for i in range(num_uncovered):
            idx = indices[i]
            x, y = uncovered_x[idx], uncovered_y[idx]

            color_idx = np.random.choice(len(colors), p=probs)
            color = colors[color_idx]

            canvas[y, x, :] = color
            coverage_mask[y, x] = True
            budget_feedback.update_usage(color_idx, 1)

            if i % 1000 == 0 and i > 0:
                remaining_budget = budget_feedback.get_remaining_budget()
                total_remaining = np.sum(remaining_budget)

                if total_remaining > 0:
                    probs = remaining_budget / total_remaining
                else:
                    probs = budget_feedback.current_usage / np.sum(budget_feedback.current_usage)
                probs = probs / np.sum(probs)

        if np.any(~coverage_mask):
            final_uncovered = np.where(~coverage_mask)
            for y, x in zip(final_uncovered[0], final_uncovered[1]):
                color_idx = np.argmax(budget_feedback.current_usage)
                canvas[y, x, :] = colors[color_idx]
                coverage_mask[y, x] = True
                budget_feedback.update_usage(color_idx, 1)

        print(f"像素级精修完成：填充了 {num_uncovered} 个像素")
        return canvas, coverage_mask

    def select_color_by_budget(self, budget_feedback: BudgetFeedback) -> int:
        """
        根据剩余预算选择颜色

        参数:
            budget_feedback: 预算反馈对象
        返回:
            color_idx: 选中的颜色索引
        """
        remaining_budget = budget_feedback.get_remaining_budget()
        total_remaining = np.sum(remaining_budget)

        if total_remaining > 0:
            probs = remaining_budget / total_remaining
            return np.random.choice(len(probs), p=probs)
        else:
            return np.random.randint(len(budget_feedback.target_ratios))

    def validate_and_adjust_budgets(self, target_ratios: List[float]) -> List[float]:
        """
        验证和调整颜色预算（支持可变颜色数量）

        参数:
            target_ratios: 目标比例列表
        返回:
            adjusted_ratios: 调整后的比例列表（总和为 1）
        """
        adjusted_ratios = target_ratios.copy()
        n_colors = len(target_ratios)
        min_budget_threshold = 0.05

        under_budget_indices = [i for i, ratio in enumerate(target_ratios)
                                if ratio < min_budget_threshold]

        if under_budget_indices:
            total_under = sum(min_budget_threshold - target_ratios[i]
                              for i in under_budget_indices)

            over_budget_indices = [i for i in range(n_colors)
                                   if i not in under_budget_indices]

            if over_budget_indices:
                over_ratios = [target_ratios[i] for i in over_budget_indices]
                total_over = sum(over_ratios)

                if total_over > 0:
                    deduction_ratios = [total_under * (over_ratios[j] / total_over)
                                        for j in range(len(over_ratios))]
                    adjusted_over = [over_ratios[j] - deduction_ratios[j]
                                     for j in range(len(over_ratios))]

                    for idx, val in zip(over_budget_indices, adjusted_over):
                        adjusted_ratios[idx] = max(val, min_budget_threshold)

            for idx in under_budget_indices:
                adjusted_ratios[idx] = min_budget_threshold

            total = sum(adjusted_ratios)
            if total > 0:
                adjusted_ratios = [r / total for r in adjusted_ratios]

        return adjusted_ratios