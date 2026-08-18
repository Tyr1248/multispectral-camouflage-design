import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import warnings
from dataclasses import dataclass


@dataclass
class BudgetFeedback:
    """Color budget feedback control data structure (supports variable color counts)."""
    target_ratios: np.ndarray
    current_usage: np.ndarray
    total_placed: int
    target_pixels: np.ndarray

    @property
    def usage_ratios(self) -> np.ndarray:
        """Current usage ratios."""
        if self.total_placed == 0:
            return np.zeros_like(self.target_ratios)
        return self.current_usage / self.total_placed

    @property
    def deviation_ratios(self) -> np.ndarray:
        """Deviation rate from the target ratios."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return (self.usage_ratios - self.target_ratios) / (self.target_ratios + 1e-10)

    @property
    def remaining_budget_ratios(self) -> np.ndarray:
        """Remaining budget share (relative to the total remaining budget of all colors)."""
        remaining = self.get_remaining_budget()
        total_remaining = np.sum(remaining)
        if total_remaining < 1e-10:
            return np.zeros_like(remaining)
        return remaining / total_remaining

    def update_usage(self, color_idx: int, added_pixels: int):
        """Update color usage."""
        self.current_usage[color_idx] += added_pixels
        self.total_placed += added_pixels

    def get_remaining_budget(self) -> np.ndarray:
        """Get the remaining budget (target pixels - current usage)."""
        return np.maximum(self.target_pixels - self.current_usage, 0)

    def get_remaining_budget_ratio(self, color_idx: int) -> float:
        """Return the remaining budget share for one color."""
        return self.remaining_budget_ratios[color_idx]


class DigitalCamouflageRandom:
    """Generate random digital camouflage patterns (4x upscaled output version)."""

    def __init__(self, canvas_size: Tuple[int, int] = (256, 256),
                 spot_database_path: str = 'spot_database',
                 expand_pixels: int = 10,
                 upscale_factor: int = 4):
        """
        Initialize the camouflage generator.

        Args:
            canvas_size: Initial canvas size (width, height), default (256, 256).
            spot_database_path: Path to the spot database.
            expand_pixels: Number of edge-expansion pixels, default 10.
            upscale_factor: Upscaling factor, default 4 (output size = canvas_size x upscale_factor).
        """
        self.original_canvas_size = canvas_size
        self.expanded_canvas_size = (canvas_size[0] + expand_pixels * 2,
                                     canvas_size[1] + expand_pixels * 2)
        self.expand_pixels = expand_pixels
        self.upscale_factor = upscale_factor

        # Upscaled dimensions
        self.upscaled_canvas_size = (canvas_size[0] * upscale_factor,
                                     canvas_size[1] * upscale_factor)
        self.upscaled_expand_pixels = expand_pixels * upscale_factor

        # Initialize the spot database manager
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
        Generate a camouflage pattern (upscaled output version).

        Args:
            dominant_colors_rgb: Dominant color list, shape (n_colors, 3), dtype=np.uint8.
            total_coverage_target: Total coverage target, 0.0-1.0.
            target_ratios: Target ratio list per color, length=n_colors, auto-normalized.
            seed: Random seed, optional.

        Returns:
            canvas: Camouflage pattern image, shape (H*factor, W*factor, 3), dtype=np.uint8,
                   where H,W are canvas_size and factor is upscale_factor.
                   Example: canvas_size=(256,256), upscale_factor=4 -> output (1024, 1024, 3)
        """
        if seed is not None:
            np.random.seed(seed)

        # Validate and adjust the color budgets
        target_ratios = self.validate_and_adjust_budgets(target_ratios)

        width, height = self.expanded_canvas_size
        n_colors = len(dominant_colors_rgb)

        # Create a blank canvas (low resolution)
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        coverage_mask = np.zeros((height, width), dtype=bool)

        # Initialize the color budget feedback controller
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

        # Stage 1: grid-based base placement (budget-to-spot-size binding strategy)
        print("\n阶段 1: 网格化基础放置（预算 - 斑点大小绑定）")
        canvas, coverage_mask = self.grid_based_placement(
            canvas, coverage_mask, dominant_colors_rgb, budget_feedback,
            coverage_target=0.7
        )

        # Stage 2: gap filling (small spots only)
        print("\n阶段 2: 间隙填充（全部小斑点）")
        canvas, coverage_mask = self.gap_filling(
            canvas, coverage_mask, dominant_colors_rgb, budget_feedback,
            coverage_target=0.9
        )

        # Stage 3: pixel-level refinement (ensures 100% coverage)
        print("\n阶段 3: 像素级精修")
        canvas, coverage_mask = self.pixel_level_refinement(
            canvas, coverage_mask, dominant_colors_rgb, budget_feedback
        )

        # Verify the coverage
        current_coverage = np.sum(coverage_mask) / total_pixels
        final_usage_ratios = budget_feedback.usage_ratios

        print(f"\n低分辨率生成完成:")
        print(f"  覆盖率：{current_coverage * 100:.1f}%")
        print(f"  颜色比例：{final_usage_ratios}")
        print(f"  最大偏差：{np.max(np.abs(final_usage_ratios - target_ratios)) * 100:.1f}%")

        # Crop back to the original size (low resolution)
        from Camo.image_processor import ImageProcessor
        canvas = ImageProcessor.crop_to_original_size(
            canvas, self.expand_pixels, self.original_canvas_size
        )

        # Upscale to the final size
        print(f"\n阶段 4: 上采样 {self.upscale_factor}x")
        canvas = self.upscale_image(canvas)
        print(f"  输出尺寸：{canvas.shape[1]}×{canvas.shape[0]}")

        return canvas

    def upscale_image(self, canvas: np.ndarray) -> np.ndarray:
        """
        Upsample the image with nearest-neighbor interpolation.
        Keeps color-block edges sharp without blurring.

        Args:
            canvas: Input image, shape (H, W, 3), dtype=np.uint8.
        Returns:
            upscaled: Upsampled image, shape (H*factor, W*factor, 3), dtype=np.uint8.
        """
        h, w, c = canvas.shape
        factor = self.upscale_factor

        # Nearest-neighbor upsampling via np.repeat
        upscaled = np.repeat(canvas, factor, axis=0)  # rows x factor
        upscaled = np.repeat(upscaled, factor, axis=1)  # columns x factor

        return upscaled

    def select_spot_type_by_budget(self, budget_feedback: BudgetFeedback,
                                   color_idx: int) -> str:
        """
        Select the spot type based on the color budget share.

        Budget share rules:
        - > 0.6: 100% large spots
        - 0.3 ~ 0.6: mix of large and small spots (large 40% : small 60%)
        - <= 0.3: 100% small spots

        Args:
            budget_feedback: Budget feedback object.
            color_idx: Color index.
        Returns:
            spot_type: 'large' or 'small'
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
        Grid-based base placement: place spots on a regular grid for basic coverage.
        Applies the budget-to-spot-size binding strategy.

        Args:
            canvas: Canvas array, shape (H, W, 3).
            coverage_mask: Coverage mask, shape (H, W).
            colors: Color array, shape (n_colors, 3).
            budget_feedback: Budget feedback object.
            coverage_target: Target coverage ratio.
        Returns:
            canvas, coverage_mask: Updated canvas and mask.
        """
        height, width, _ = canvas.shape
        total_pixels = height * width
        target_pixels = int(total_pixels * coverage_target)

        # Compute the grid size
        avg_spot_size = 12
        grid_size = max(avg_spot_size // 2, 8)

        # Create grid points
        grid_x = np.arange(grid_size, width - grid_size, grid_size)
        grid_y = np.arange(grid_size, height - grid_size, grid_size)

        # Shuffle the grid points
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

            # Select a color based on the budget
            color_idx = self.select_color_by_budget(budget_feedback)
            color = colors[color_idx]

            # Select the spot size based on the color budget
            spot_type = self.select_spot_type_by_budget(budget_feedback, color_idx)

            # Get a spot
            spot_cell = self.spot_database_manager.spot_database.get(spot_type, [])
            if not spot_cell:
                continue

            spot_data = spot_cell[np.random.randint(len(spot_cell))]
            rotation_angle = np.random.choice([0, 90, 180, 270])
            rotated_spot_data = self.spot_renderer.rotate_spot(spot_data, rotation_angle)
            spot_img = rotated_spot_data['image']
            sh, sw = spot_img.shape

            # Make sure the spot stays within the bounds
            if x + sw >= width or y + sh >= height:
                continue

            # Place the spot
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
        Gap filling: place small spots in blank areas.
        Uses small spots only, ignoring the budget.

        Args:
            canvas: Canvas array, shape (H, W, 3).
            coverage_mask: Coverage mask, shape (H, W).
            colors: Color array, shape (n_colors, 3).
            budget_feedback: Budget feedback object.
            coverage_target: Target coverage ratio.
        Returns:
            canvas, coverage_mask: Updated canvas and mask.
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

            # Always use small spots
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
        Pixel-level refinement: ensures 100% coverage and satisfies the color budget.

        Args:
            canvas: Canvas array, shape (H, W, 3).
            coverage_mask: Coverage mask, shape (H, W).
            colors: Color array, shape (n_colors, 3).
            budget_feedback: Budget feedback object.
        Returns:
            canvas, coverage_mask: Updated canvas and mask.
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
        Select a color based on the remaining budget.

        Args:
            budget_feedback: Budget feedback object.
        Returns:
            color_idx: Selected color index.
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
        Validate and adjust the color budgets (supports variable color counts).

        Args:
            target_ratios: Target ratio list.
        Returns:
            adjusted_ratios: Adjusted ratio list (sums to 1).
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