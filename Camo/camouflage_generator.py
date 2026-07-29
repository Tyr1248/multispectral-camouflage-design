"""
迷彩图案生成器
生成非晶态和晶态GST对应的迷彩图案
"""

import os
import numpy as np
import cv2
from datetime import datetime
import json
import warnings
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from sklearn.cluster import KMeans
from skopt import Optimizer
from skopt.space import Real
import traceback

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
    """颜色空间分析器（基于环境图像）"""

    def __init__(self, color_distance_threshold: float = 20.0):
        self.color_distance_threshold = color_distance_threshold

    def analyze_colors_from_backgrounds(self, background_imgs: List[np.ndarray],
                                        input_colors_rgb: np.ndarray) -> ColorAnalysisResult:
        """从多个环境图像分析颜色"""
        print(f"从 {len(background_imgs)} 张环境图像分析主要颜色...")

        # 合并所有环境图像的主要颜色
        all_background_colors = []
        for bg_img in background_imgs:
            colors = self.extract_dominant_colors_from_background(bg_img)
            all_background_colors.append(colors)

        # 合并所有提取的颜色
        if all_background_colors:
            background_colors = np.vstack(all_background_colors)
            # 对合并的颜色再次聚类，取前6个
            background_colors = self.cluster_colors(background_colors, n_clusters=6)
        else:
            # 如果没有环境图像，使用输入颜色
            background_colors = input_colors_rgb

        print(f"从环境图像提取到 {len(background_colors)} 种主要颜色")

        # 计算颜色相似度
        similarity_matrix = self.calculate_color_similarities(input_colors_rgb, background_colors)

        # 初步确定主色
        n_dominant, dominant_indices = self.preliminary_dominant_detection(
            input_colors_rgb, similarity_matrix
        )

        return ColorAnalysisResult(
            n_dominant_colors=n_dominant,
            dominant_color_indices=dominant_indices,
            background_dominant_colors=background_colors,
            input_colors=input_colors_rgb,
            color_similarity_matrix=similarity_matrix
        )

    def extract_dominant_colors_from_background(self, background_img: np.ndarray,
                                                n_colors: int = 6) -> np.ndarray:
        """从单张环境图像提取主要颜色"""
        # 下采样以减少计算量
        height, width = background_img.shape[:2]
        max_pixels = 10000
        if height * width > max_pixels:
            scale = np.sqrt(max_pixels / (height * width))
            new_height = int(height * scale)
            new_width = int(width * scale)
            resized_img = cv2.resize(background_img, (new_width, new_height))
        else:
            resized_img = background_img

        # 重塑为像素列表
        pixels = resized_img.reshape(-1, 3)

        # 使用K-means聚类
        if len(pixels) >= n_colors:
            kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
            kmeans.fit(pixels)
            dominant_colors = kmeans.cluster_centers_.astype(np.uint8)

            # 按聚类大小排序
            labels = kmeans.labels_
            cluster_sizes = np.bincount(labels)
            sorted_indices = np.argsort(cluster_sizes)[::-1]
            dominant_colors = dominant_colors[sorted_indices]
        else:
            # 像素太少，直接采样
            indices = np.random.choice(len(pixels), min(n_colors, len(pixels)), replace=False)
            dominant_colors = pixels[indices].astype(np.uint8)

        return dominant_colors

    def cluster_colors(self, colors: np.ndarray, n_clusters: int = 6) -> np.ndarray:
        """对颜色进行聚类，减少数量"""
        if len(colors) <= n_clusters:
            return colors

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(colors)
        cluster_centers = kmeans.cluster_centers_.astype(np.uint8)
        return cluster_centers

    def calculate_color_similarities(self, colors1: np.ndarray, colors2: np.ndarray) -> np.ndarray:
        """计算颜色相似度矩阵"""
        n1 = colors1.shape[0]
        n2 = colors2.shape[0]
        similarity_matrix = np.zeros((n1, n2))

        # 转换到CIELAB空间
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

    def preliminary_dominant_detection(self, input_colors: np.ndarray,
                                       similarity_matrix: np.ndarray) -> Tuple[int, List[int]]:
        """初步确定主色"""
        n_input_colors = input_colors.shape[0]

        # 计算每个输入颜色与背景的最大相似度
        color_background_similarity = np.max(similarity_matrix, axis=1)

        # 找到与背景最相似的两个颜色
        top_indices = np.argsort(color_background_similarity)[-2:][::-1]

        if len(top_indices) < 2:
            return 1, [0]

        # 计算这两个颜色之间的距离
        colors_lab = self.rgb_to_lab(input_colors)
        color_distance = np.sqrt(np.sum((colors_lab[top_indices[0]] - colors_lab[top_indices[1]]) ** 2))

        # 根据距离确定主色数量
        if color_distance < self.color_distance_threshold:
            n_dominant = 1
            dominant_indices = [top_indices[0]]
        else:
            n_dominant = 2
            dominant_indices = list(top_indices)

        print(f"初步确定: {n_dominant} 个主色, 索引: {dominant_indices}")

        return n_dominant, dominant_indices


@dataclass
class BudgetFeedback:
    """颜色预算反馈控制数据结构"""
    target_ratios: np.ndarray
    current_usage: np.ndarray
    total_placed: int
    target_pixels: np.ndarray

    @property
    def usage_ratios(self) -> np.ndarray:
        if self.total_placed == 0:
            return np.zeros_like(self.target_ratios)
        return self.current_usage / self.total_placed

    def update_usage(self, color_idx: int, added_pixels: int):
        self.current_usage[color_idx] += added_pixels
        self.total_placed += added_pixels

    def get_remaining_budget(self) -> np.ndarray:
        return np.maximum(self.target_pixels - self.current_usage, 0)


class DigitalCamouflageGenerator:
    """数字迷彩生成器"""

    def __init__(self, canvas_size: Tuple[int, int] = (256, 256),
                 expand_pixels: int = 10):
        self.original_canvas_size = canvas_size
        self.expanded_canvas_size = (canvas_size[0] + expand_pixels * 2,
                                     canvas_size[1] + expand_pixels * 2)
        self.expand_pixels = expand_pixels

    def generate_camouflage(self, colors: np.ndarray,
                            target_ratios: List[float],
                            seed: int = None) -> np.ndarray:
        """生成迷彩图案"""
        if seed is not None:
            np.random.seed(seed)

        # 调整颜色预算
        target_ratios = self.validate_and_adjust_budgets(target_ratios)

        width, height = self.expanded_canvas_size

        # 创建画布
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        coverage_mask = np.zeros((height, width), dtype=bool)

        # 初始化预算反馈
        total_pixels = width * height
        target_covered_pixels = total_pixels  # 100%覆盖

        budget_feedback = BudgetFeedback(
            target_ratios=np.array(target_ratios),
            current_usage=np.zeros(len(target_ratios)),
            total_placed=0,
            target_pixels=np.array(target_ratios) * target_covered_pixels
        )

        # 阶段1: 网格化基础放置
        canvas, coverage_mask = self.grid_based_placement(
            canvas, coverage_mask, colors, budget_feedback, coverage_target=0.7
        )

        # 阶段2: 间隙填充
        canvas, coverage_mask = self.gap_filling(
            canvas, coverage_mask, colors, budget_feedback, coverage_target=0.9
        )

        # 阶段3: 像素级精修
        canvas, coverage_mask = self.pixel_level_refinement(
            canvas, coverage_mask, colors, budget_feedback
        )

        # 裁切回原始尺寸
        canvas = self.crop_to_original_size(canvas)

        return canvas

    def grid_based_placement(self, canvas, coverage_mask, colors,
                             budget_feedback: BudgetFeedback, coverage_target: float):
        """网格化基础放置"""
        height, width, _ = canvas.shape
        total_pixels = height * width
        target_pixels = int(total_pixels * coverage_target)

        # 创建网格
        grid_size = 12
        grid_x = np.arange(grid_size, width - grid_size, grid_size)
        grid_y = np.arange(grid_size, height - grid_size, grid_size)

        # 打乱网格点
        grid_positions = [(x, y) for x in grid_x for y in grid_y]
        np.random.shuffle(grid_positions)

        placed_pixels = 0

        # 简单的斑点模拟（使用圆形斑点）
        for x, y in grid_positions:
            if placed_pixels >= target_pixels:
                break

            # 随机选择斑点大小
            spot_size = np.random.randint(8, 20)

            # 根据预算选择颜色
            color_idx = self.select_color_by_budget(budget_feedback)
            color = colors[color_idx]

            # 放置圆形斑点
            added = 0
            for i in range(-spot_size // 2, spot_size // 2 + 1):
                for j in range(-spot_size // 2, spot_size // 2 + 1):
                    if i * i + j * j <= (spot_size // 2) ** 2:
                        px = x + i
                        py = y + j

                        if 0 <= px < width and 0 <= py < height:
                            if not coverage_mask[py, px]:
                                canvas[py, px, :] = color
                                coverage_mask[py, px] = True
                                added += 1

            if added > 0:
                budget_feedback.update_usage(color_idx, added)
                placed_pixels += added

        return canvas, coverage_mask

    def gap_filling(self, canvas, coverage_mask, colors,
                    budget_feedback: BudgetFeedback, coverage_target: float):
        """间隙填充"""
        height, width, _ = canvas.shape
        total_pixels = height * width
        target_pixels = int(total_pixels * coverage_target)

        placed_pixels = np.sum(coverage_mask)
        max_attempts = 10000

        for _ in range(max_attempts):
            if placed_pixels >= target_pixels:
                break

            # 找到空白区域
            uncovered_y, uncovered_x = np.where(~coverage_mask)
            if len(uncovered_y) == 0:
                break

            # 随机选择空白点
            idx = np.random.randint(len(uncovered_y))
            x, y = uncovered_x[idx], uncovered_y[idx]

            # 小斑点
            spot_size = np.random.randint(4, 8)
            color_idx = self.select_color_by_budget(budget_feedback)
            color = colors[color_idx]

            added = 0
            for i in range(-spot_size // 2, spot_size // 2 + 1):
                for j in range(-spot_size // 2, spot_size // 2 + 1):
                    if i * i + j * j <= (spot_size // 2) ** 2:
                        px = x + i
                        py = y + j

                        if 0 <= px < width and 0 <= py < height:
                            if not coverage_mask[py, px]:
                                canvas[py, px, :] = color
                                coverage_mask[py, px] = True
                                added += 1

            if added > 0:
                budget_feedback.update_usage(color_idx, added)
                placed_pixels += added

        return canvas, coverage_mask

    def pixel_level_refinement(self, canvas, coverage_mask, colors,
                               budget_feedback: BudgetFeedback):
        """像素级精修"""
        height, width, _ = canvas.shape

        # 找到未覆盖像素
        uncovered_indices = np.where(~coverage_mask)
        uncovered_y, uncovered_x = uncovered_indices
        num_uncovered = len(uncovered_y)

        if num_uncovered == 0:
            return canvas, coverage_mask

        # 打乱顺序
        indices = np.random.permutation(num_uncovered)

        # 计算颜色概率
        remaining_budget = budget_feedback.get_remaining_budget()
        total_remaining = np.sum(remaining_budget)

        if total_remaining > 0:
            probs = remaining_budget / total_remaining
        else:
            probs = budget_feedback.current_usage / np.sum(budget_feedback.current_usage)

        probs = probs / np.sum(probs)

        # 填充像素
        for i in range(num_uncovered):
            idx = indices[i]
            x, y = uncovered_x[idx], uncovered_y[idx]

            color_idx = np.random.choice(len(colors), p=probs)
            color = colors[color_idx]

            canvas[y, x, :] = color
            coverage_mask[y, x] = True
            budget_feedback.update_usage(color_idx, 1)

        # 强制覆盖剩余像素
        if np.any(~coverage_mask):
            final_uncovered = np.where(~coverage_mask)
            for y, x in zip(final_uncovered[0], final_uncovered[1]):
                color_idx = np.argmax(budget_feedback.current_usage)
                canvas[y, x, :] = colors[color_idx]
                coverage_mask[y, x] = True

        return canvas, coverage_mask

    def select_color_by_budget(self, budget_feedback: BudgetFeedback) -> int:
        """根据预算选择颜色"""
        remaining_budget = budget_feedback.get_remaining_budget()
        total_remaining = np.sum(remaining_budget)

        if total_remaining > 0:
            probs = remaining_budget / total_remaining
            return np.random.choice(len(probs), p=probs)
        else:
            return np.random.randint(len(budget_feedback.target_ratios))

    def validate_and_adjust_budgets(self, target_ratios: List[float]) -> List[float]:
        """验证和调整颜色预算"""
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

    def crop_to_original_size(self, canvas: np.ndarray) -> np.ndarray:
        """裁切回原始尺寸"""
        if self.expand_pixels == 0:
            return canvas

        height, width = canvas.shape[:2]
        start_x = self.expand_pixels
        start_y = self.expand_pixels
        end_x = width - self.expand_pixels
        end_y = height - self.expand_pixels

        return canvas[start_y:end_y, start_x:end_x, :]


class BayesianOptimizer:
    """贝叶斯优化器"""

    def __init__(self, n_seeds: int = 3, n_samples_per_seed: int = 2):
        self.n_seeds = n_seeds
        self.n_samples_per_seed = n_samples_per_seed
        self.history = []

    def optimize_budget(self,
                        amorphous_colors: np.ndarray,
                        color_analysis_result: ColorAnalysisResult,
                        environment_blocks: List[np.ndarray],
                        n_calls: int = 20) -> np.ndarray:
        """优化颜色预算"""
        n_colors = amorphous_colors.shape[0]
        n_dominant = color_analysis_result.n_dominant_colors
        dominant_indices = color_analysis_result.dominant_color_indices

        print(f"贝叶斯优化:")
        print(f"  主色数量: {n_dominant}")
        print(f"  主色索引: {dominant_indices}")

        # 定义搜索空间
        dimensions = self.define_search_space(n_dominant)

        # 创建优化器
        optimizer = Optimizer(
            dimensions=dimensions['space'],
            n_initial_points=5,
            acq_func='EI',
            random_state=42
        )

        # 运行优化
        for i in range(n_calls):
            next_params = optimizer.ask()

            y_val = self.objective(next_params, dimensions['names'], n_dominant,
                                   dominant_indices, n_colors, amorphous_colors,
                                   environment_blocks)

            optimizer.tell(next_params, y_val)

            if (i + 1) % 5 == 0:
                best_idx = np.argmin(optimizer.yi)
                best_score = -optimizer.yi[best_idx]
                print(f"  迭代 {i + 1}/{n_calls}: 当前最佳得分 = {best_score:.4f}")

        # 提取最佳结果
        best_idx = np.argmin(optimizer.yi)
        best_params = optimizer.Xi[best_idx]

        best_budget = self.params_to_budget(
            dict(zip(dimensions['names'], best_params)),
            n_dominant, dominant_indices, n_colors
        )

        return best_budget

    def define_search_space(self, n_dominant: int):
        """定义搜索空间"""
        dimensions = {'names': [], 'space': []}

        if n_dominant == 1:
            dimensions['names'].append('main_ratio')
            dimensions['space'].append(Real(0.3, 0.8, name='main_ratio'))
        else:
            dimensions['names'].append('main_ratio_1')
            dimensions['space'].append(Real(0.2, 0.6, name='main_ratio_1'))
            dimensions['names'].append('main_ratio_2')
            dimensions['space'].append(Real(0.2, 0.6, name='main_ratio_2'))

        return dimensions

    def params_to_budget(self, params: Dict[str, float],
                         n_dominant: int, dominant_indices: List[int],
                         n_colors: int) -> np.ndarray:
        """参数转换为预算"""
        budget = np.zeros(n_colors)

        if n_dominant == 1:
            main_ratio = params['main_ratio']
            main_idx = dominant_indices[0]

            budget[main_idx] = main_ratio
            remaining = 1.0 - main_ratio
            other_colors = [i for i in range(n_colors) if i != main_idx]

            if other_colors:
                other_ratio = remaining / len(other_colors)
                for idx in other_colors:
                    budget[idx] = other_ratio
        else:
            ratio1 = params['main_ratio_1']
            ratio2 = params['main_ratio_2']
            idx1, idx2 = dominant_indices[0], dominant_indices[1]

            if ratio1 + ratio2 > 0.8:
                scale = 0.8 / (ratio1 + ratio2)
                ratio1 *= scale
                ratio2 *= scale

            budget[idx1] = ratio1
            budget[idx2] = ratio2

            remaining = 1.0 - ratio1 - ratio2
            other_colors = [i for i in range(n_colors) if i not in [idx1, idx2]]

            if other_colors:
                other_ratio = remaining / len(other_colors)
                for idx in other_colors:
                    budget[idx] = other_ratio

        budget = budget / np.sum(budget)
        return budget

    def objective(self, params_list, param_names, n_dominant, dominant_indices,
                  n_colors, amorphous_colors, environment_blocks):
        """目标函数"""
        params = dict(zip(param_names, params_list))

        budget = self.params_to_budget(params, n_dominant, dominant_indices, n_colors)

        # 评估预算
        score = self.evaluate_budget(budget, amorphous_colors, environment_blocks)

        self.history.append({
            'params': params.copy(),
            'budget': budget.copy(),
            'score': score
        })

        return -score

    def evaluate_budget(self, budget: np.ndarray,
                        amorphous_colors: np.ndarray,
                        environment_blocks: List[np.ndarray]) -> float:
        """评估预算"""
        from similarity_metrics import ssim_rgb, uiqi

        scores = []

        for seed in range(self.n_seeds):
            # 生成迷彩
            generator = DigitalCamouflageGenerator(canvas_size=(256, 256), expand_pixels=10)

            camouflage = generator.generate_camouflage(
                amorphous_colors, budget.tolist(), seed=seed
            )

            # 评估与所有环境块的相似度
            block_scores = []
            for block in environment_blocks:
                ssim_val = ssim_rgb(camouflage, block)
                # uiqi_val = uiqi(camouflage, block)
                combined = 0.6 * ssim_val
                block_scores.append(combined)

            avg_score = np.mean(block_scores)
            scores.append(avg_score)

        return np.mean(scores)


def sample_environment_blocks(environment_img: np.ndarray, n_samples: int = 4,
                              block_size: int = 256) -> List[np.ndarray]:
    """
    从环境图像中随机采样区块

    Args:
        environment_img: 环境图像 (H×W×3)
        n_samples: 采样数量
        block_size: 区块大小

    Returns:
        区块列表
    """
    height, width = environment_img.shape[:2]

    if height < block_size or width < block_size:
        raise ValueError(f"环境图像尺寸({width}x{height})小于区块大小({block_size}x{block_size})")

    blocks = []

    for _ in range(n_samples):
        # 随机选择起始位置
        x = np.random.randint(0, width - block_size)
        y = np.random.randint(0, height - block_size)

        # 提取区块
        block = environment_img[y:y + block_size, x:x + block_size, :]
        blocks.append(block)

    return blocks


def generate_camouflage_pattern(
        amorphous_colors: List[List[int]],
        crystalline_colors: List[List[int]],
        environment_paths: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    根据输入的非晶态颜色和晶态颜色生成迷彩图案

    Args:
        amorphous_colors: 非晶态颜色列表 [[R,G,B], ...]
        crystalline_colors: 晶态颜色列表 [[R,G,B], ...]
        environment_paths: 环境图像路径列表

    Returns:
        包含迷彩图案和信息的字典
    """
    print("=" * 60)
    print("迷彩图案生成器启动")
    print("=" * 60)

    # 转换颜色为numpy数组
    amorphous_colors_np = np.array(amorphous_colors, dtype=np.uint8)
    crystalline_colors_np = np.array(crystalline_colors, dtype=np.uint8)

    # 验证颜色数量
    if len(amorphous_colors) != len(crystalline_colors):
        raise ValueError("非晶态颜色和晶态颜色数量必须相同")

    n_colors = len(amorphous_colors)
    print(f"颜色数量: {n_colors}")
    print(f"非晶态颜色: {amorphous_colors}")
    print(f"晶态颜色: {crystalline_colors}")

    # 加载环境图像（如果有）
    environment_blocks = []
    environment_used = False

    if environment_paths and len(environment_paths) > 0:
        environment_used = True
        print(f"\n加载环境图像 ({len(environment_paths)}张)...")

        all_environment_imgs = []

        for env_path in environment_paths:
            if not os.path.exists(env_path):
                print(f"警告: 环境图像不存在: {env_path}")
                continue

            # 读取图像
            env_img = cv2.imread(env_path)
            if env_img is None:
                print(f"警告: 无法读取环境图像: {env_path}")
                continue

            # 转换为RGB
            env_img_rgb = cv2.cvtColor(env_img, cv2.COLOR_BGR2RGB)

            # 验证图像尺寸
            height, width = env_img_rgb.shape[:2]
            if height < 512 or width < 512:
                print(f"警告: 环境图像({width}x{height})尺寸过小，建议使用512x512以上图像")

            all_environment_imgs.append(env_img_rgb)

            # 从环境图像中采样区块
            try:
                blocks = sample_environment_blocks(env_img_rgb, n_samples=4, block_size=256)
                environment_blocks.extend(blocks)
                print(f"  从 {os.path.basename(env_path)} 采样4个256x256区块")
            except ValueError as e:
                print(f"  警告: {e}")

        if len(all_environment_imgs) == 0:
            print("警告: 未成功加载任何环境图像，将使用默认预算")
            environment_used = False
        else:
            print(f"总共采样 {len(environment_blocks)} 个环境区块")

    # 确定颜色预算
    if environment_used and len(environment_blocks) > 0:
        print("\n--- 颜色空间分析 ---")
        color_analyzer = ColorAnalyzer(color_distance_threshold=20.0)

        # 分析环境颜色
        color_analysis = color_analyzer.analyze_colors_from_backgrounds(
            all_environment_imgs, amorphous_colors_np
        )

        print("\n--- 贝叶斯优化颜色预算 ---")
        optimizer = BayesianOptimizer(n_seeds=3, n_samples_per_seed=2)

        try:
            best_budget = optimizer.optimize_budget(
                amorphous_colors_np, color_analysis, environment_blocks, n_calls=15
            )

            print(f"\n优化完成!")
            print(f"最佳颜色预算: {best_budget}")

        except Exception as e:
            print(f"贝叶斯优化失败: {e}")
            print("使用默认预算: 平均分配")
            best_budget = np.ones(n_colors) / n_colors
    else:
        print("\n未使用环境图像，使用默认预算: 平均分配")
        best_budget = np.ones(n_colors) / n_colors

    # 生成迷彩图案
    print("\n--- 生成迷彩图案 ---")

    # 生成非晶态迷彩
    print("生成非晶态迷彩...")
    generator = DigitalCamouflageGenerator(canvas_size=(256, 256), expand_pixels=10)

    amorphous_pattern = generator.generate_camouflage(
        amorphous_colors_np, best_budget.tolist(), seed=2023
    )

    # 生成晶态迷彩（使用相同的预算，但替换颜色）
    print("生成晶态迷彩...")
    # 注意：我们使用相同的生成器，但传入晶态颜色
    crystalline_pattern = generator.generate_camouflage(
        crystalline_colors_np, best_budget.tolist(), seed=2023
    )

    print("\n迷彩图案生成完成!")

    # 返回结果
    return {
        'amorphous_pattern': amorphous_pattern,
        'crystalline_pattern': crystalline_pattern,
        'pattern_size': (256, 256),
        'color_count': n_colors,
        'environment_used': environment_used,
        'color_budget': best_budget.tolist() if environment_used else None
    }