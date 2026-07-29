import cv2
import numpy as np
from Cluster_extraction.KHM import OptimizedKHM
from sklearn.cluster import KMeans
import os


class ColorFeatureExtractor:
    """
    基于两阶段聚类的颜色特征提取器

    方法:
    1. 第一阶段: 对每张图像独立进行KHM聚类，得到N个颜色中心及其在图像中的占比
    2. 第二阶段: 合并所有颜色中心，使用KMeans++进行全局聚类
    3. 权重传递: 基于第一阶段颜色中心的占比计算最终颜色的代表性权重
    4. 排序输出: 按权重降序排列颜色

    参数:
        n_colors: 提取的颜色数量 (默认4)
        use_two_stage: 是否使用两阶段聚类 (默认自动判断)
        show_progress: 是否显示进度信息 (默认True)
    """

    def __init__(self, n_colors=4, use_two_stage=None, show_progress=True):
        self.n_colors = n_colors
        self.use_two_stage = use_two_stage
        self.show_progress = show_progress


    def extract_colors(self, image_paths):
        """
        从图像中提取主颜色及其权重。
        参数:
            image_paths: 字符串或字符串列表
        返回:
            列表，每个元素为 [颜色列表, 权重列表]。
            颜色列表: RGB元组列表 [(R,G,B), ...]
            权重列表: 对应颜色的占比或代表性权重（浮点数列表）
        """
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        if self.use_two_stage is None:
            use_two_stage = len(image_paths) > 1
        else:
            use_two_stage = self.use_two_stage

        if use_two_stage:
            return self._extract_colors_two_stage_with_weight_transfer(image_paths)
        else:
            return self._extract_colors_single_stage_with_proportion(image_paths)

    def _read_image_with_chinese_path(self, img_path):
        """
        支持中文路径的图像读取函数

        方法:
        1. 使用cv2.imdecode从二进制数据解码
        2. 尝试编码转换(GBK转UTF-8)
        3. 直接读取(最后尝试)
        """
        # 方法1: 使用cv2.imdecode读取
        try:
            with open(img_path, 'rb') as f:
                img_data = np.frombuffer(f.read(), dtype=np.uint8)
                img_bgr = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                if img_bgr is not None:
                    return img_bgr
        except Exception:
            pass

        # 方法2: 如果文件存在但读取失败，尝试编码转换
        if os.path.exists(img_path):
            try:
                encoded_path = img_path.encode('gbk').decode('utf-8', errors='ignore')
                img_bgr = cv2.imread(encoded_path)
                if img_bgr is not None:
                    return img_bgr
            except:
                pass

            # 方法3: 直接读取
            img_bgr = cv2.imread(img_path)
            if img_bgr is not None:
                return img_bgr

        return None

    def _extract_colors_single_stage_with_proportion(self, image_paths):
        """
        单阶段聚类: 对每个图像独立提取主颜色，按占比排序

        算法:
        1. 对每张图像进行KHM聚类，得到N个颜色中心
        2. 计算每个颜色中心在图像中的占比
        3. 按占比降序排列颜色

        返回:
            每个图像的颜色列表列表
        """
        all_extracted = []
        total_images = len(image_paths)

        for idx, img_path in enumerate(image_paths):
            if self.show_progress:
                print(f"处理图像 {idx + 1}/{total_images}: {os.path.basename(img_path)}")

            # 读取图像
            img_bgr = self._read_image_with_chinese_path(img_path)
            if img_bgr is None:
                print(f"警告: 无法读取图像 {img_path}")
                continue

            # 转换为Lab颜色空间
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)

            # 像素采样
            pixels = img_lab.reshape(-1, 3).astype(np.float32)

            # KHM聚类
            if self.show_progress:
                print("  正在进行KHM聚类...")

            khm = OptimizedKHM(n_clusters=self.n_colors)
            khm.fit(pixels)
            lab_colors = khm.cluster_centers_

            # 计算占比
            proportions = self._calculate_color_proportions(lab_colors, pixels)
            # 按占比降序排序
            sorted_indices = np.argsort(-proportions)
            sorted_lab = lab_colors[sorted_indices]
            sorted_props = proportions[sorted_indices]

            rgb_colors = self._lab_to_rgb_tuples(sorted_lab)
            all_extracted.append([rgb_colors, sorted_props.tolist()])
        return all_extracted

    def _extract_colors_two_stage_with_weight_transfer(self, image_paths):
        """
        两阶段聚类: 基于权重传递的颜色提取

        算法:
        第一阶段:
        1. 对每张图像进行KHM聚类，得到N个颜色中心
        2. 计算每个颜色中心在图像中的占比

        第二阶段:
        1. 合并所有颜色中心，进行KMeans++全局聚类
        2. 基于权重传递计算最终颜色的代表性权重
        3. 按权重降序排列颜色

        返回:
            融合后的颜色列表
        """
        if self.show_progress:
            print("=" * 60)
            print("第一阶段: 单图像KHM聚类与权重计算")
            print("=" * 60)

        # 第一阶段结果存储
        stage1_results = []  # 存储(颜色中心, 权重)对

        total_images = len(image_paths)

        for idx, img_path in enumerate(image_paths):
            if self.show_progress:
                print(f"\n处理图像 {idx + 1}/{total_images}: {os.path.basename(img_path)}")

            # 读取图像
            img_bgr = self._read_image_with_chinese_path(img_path)
            if img_bgr is None:
                print(f"警告: 无法读取图像 {img_path}")
                continue

            # 转换为Lab颜色空间
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)

            # 像素采样
            pixels = img_lab.reshape(-1, 3).astype(np.float32)

            # KHM聚类
            if self.show_progress:
                print("  正在进行KHM聚类...")

            khm = OptimizedKHM(n_clusters=self.n_colors)
            khm.fit(pixels)
            lab_colors = khm.cluster_centers_

            # 计算每个颜色中心的占比
            proportions = self._calculate_color_proportions(lab_colors, pixels)

            # 存储第一阶段结果
            for lab_color, proportion in zip(lab_colors, proportions):
                # 归一化权重：每个图像的权重总和为1
                normalized_weight = proportion / total_images
                stage1_results.append({
                    'lab_color': lab_color,
                    'weight': normalized_weight,
                    'image_idx': idx
                })

            if self.show_progress:
                print(f"  完成，得到 {len(lab_colors)} 个颜色中心")
                print(f"  颜色中心权重: {proportions}")

        if not stage1_results:
            print("错误: 没有成功读取任何图像")
            return [[]]

        if self.show_progress:
            print("\n" + "=" * 60)
            print(f"第一阶段完成，共收集 {len(stage1_results)} 个颜色中心")
            print("=" * 60)

            print("\n第二阶段: 全局KMeans++聚类与权重传递")
            print("=" * 60)

        # 提取所有颜色中心
        all_lab_colors = np.array([result['lab_color'] for result in stage1_results])

        # 第二阶段：全局KMeans++
        all_lab_colors = np.array([r['lab_color'] for r in stage1_results])
        kmeans = KMeans(n_clusters=self.n_colors, init='k-means++', n_init=10, random_state=42)
        kmeans.fit(all_lab_colors)
        final_lab_colors = kmeans.cluster_centers_

        # 权重传递
        final_weights = np.zeros(self.n_colors)
        for result in stage1_results:
            lab_color = result['lab_color']
            weight = result['weight']
            distances = np.linalg.norm(final_lab_colors - lab_color, axis=1)
            nearest_idx = np.argmin(distances)
            final_weights[nearest_idx] += weight

        # 按权重降序排序
        sorted_indices = np.argsort(-final_weights)
        sorted_lab = final_lab_colors[sorted_indices]
        sorted_weights = final_weights[sorted_indices]

        rgb_colors = self._lab_to_rgb_tuples(sorted_lab)
        return [[rgb_colors, sorted_weights.tolist()]]  # 注意外层列表

    def _calculate_color_proportions(self, lab_colors, pixels):
        """
        计算每个颜色中心在像素集合中的占比

        参数:
            lab_colors: Lab颜色数组，形状为(n_colors, 3)
            pixels: 像素数组，形状为(n_pixels, 3)

        返回:
            每个颜色中心的占比数组，形状为(n_colors,)
        """
        # 计算每个像素到每个颜色中心的距离
        distances = np.sqrt(((pixels[:, np.newaxis, :] - lab_colors) ** 2).sum(axis=2))

        # 找到每个像素的最近颜色中心
        nearest_centers = np.argmin(distances, axis=1)

        # 统计每个颜色中心的像素数量
        center_counts = np.bincount(nearest_centers, minlength=len(lab_colors))

        # 计算占比
        proportions = center_counts / len(pixels)

        return proportions

    def _calculate_and_sort_colors_by_proportion(self, lab_colors, pixels):
        """
        计算颜色占比并按占比降序排序

        参数:
            lab_colors: Lab颜色数组，形状为(n_colors, 3)
            pixels: 像素数组，形状为(n_pixels, 3)

        返回:
            按占比排序的RGB颜色元组列表
        """
        # 计算占比
        proportions = self._calculate_color_proportions(lab_colors, pixels)

        # 按占比降序排序
        sorted_indices = np.argsort(-proportions)
        sorted_lab_colors = lab_colors[sorted_indices]

        # 转换为RGB
        sorted_rgb_colors = self._lab_to_rgb_tuples(sorted_lab_colors)

        return sorted_rgb_colors

    def _lab_to_rgb_tuples(self, lab_colors):
        """
        将Lab颜色数组转换为RGB元组列表

        参数:
            lab_colors: Lab颜色数组，形状为(n_colors, 3)

        返回:
            RGB颜色元组列表: [(R,G,B), (R,G,B), ...]
        """
        rgb_tuples = []

        for lab_color in lab_colors:
            # 重塑为1x1图像以进行转换
            lab_array = lab_color.reshape(1, 1, 3).astype(np.float32)

            # 确保Lab值在正确范围内
            # OpenCV期望的Lab范围: L[0,100], a[-128,127], b[-128,127]
            # 如果数据在0-255范围，需要转换
            if np.max(lab_array) > 100:
                # 转换到标准Lab范围
                lab_array[..., 0] = lab_array[..., 0] * 100 / 255  # L通道
                lab_array[..., 1] = lab_array[..., 1] - 128  # a通道
                lab_array[..., 2] = lab_array[..., 2] - 128  # b通道

            # Lab转RGB
            rgb_array = cv2.cvtColor(lab_array, cv2.COLOR_LAB2RGB)
            # 获取RGB值（0-1范围）
            rgb_color = rgb_array[0, 0]
            # 转换为0-255范围并确保在合理范围内
            rgb_255 = np.clip(rgb_color * 255, 0, 255).astype(int)
            # 转换为元组并添加到列表
            rgb_tuples.append(tuple(rgb_255))

        return rgb_tuples

    def _format_colors_for_output(self, results):
        output = []
        for idx, (colors, weights) in enumerate(results):
            if len(results) == 1 and len(colors) == self.n_colors:
                output.append("融合主颜色 (按代表性权重排序):")
            else:
                output.append(f"图像 {idx+1}:")
            for i, (c, w) in enumerate(zip(colors, weights)):
                output.append(f"  颜色 {i+1}: RGB{c}, 权重: {w:.4f}")
        return "\n".join(output)


# 使用示例
if __name__ == "__main__":
    # 创建颜色提取器实例
    extractor = ColorFeatureExtractor(
        n_colors=4,  # 提取4种颜色
        show_progress=True  # 显示进度信息
    )

    # 示例1: 单图像提取
    print("=" * 60)
    print("示例1: 单图像颜色提取")
    print("=" * 60)

    single_image_path = "E:\ProjectX\Evaluation of Camo\image1.png"
    single_colors = extractor.extract_colors(single_image_path)
    print(single_colors)
    print(extractor._format_colors_for_output(single_colors))

    # # 示例2: 多图像融合提取
    # print("\n" + "=" * 60)
    # print("示例2: 多图像融合颜色提取")
    # print("=" * 60)
    #
    # multiple_image_paths = [
    #     "path/to/image1.jpg",
    #     "path/to/image2.jpg",
    #     "path/to/image3.jpg"
    # ]
    #
    # fused_colors = extractor.extract_colors(multiple_image_paths)
    # print(extractor._format_colors_for_output(fused_colors))
    #
    # # 示例3: 强制使用单阶段聚类处理多图像
    # print("\n" + "=" * 60)
    # print("示例3: 多图像独立颜色提取")
    # print("=" * 60)
    #
    # extractor_independent = ColorFeatureExtractor(
    #     n_colors=4,
    #     use_two_stage=False,  # 强制使用单阶段聚类
    #     show_progress=True
    # )
    #
    # independent_colors = extractor_independent.extract_colors(multiple_image_paths)
    # print(extractor_independent._format_colors_for_output(independent_colors))