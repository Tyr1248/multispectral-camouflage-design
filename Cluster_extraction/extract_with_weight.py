import cv2
import numpy as np
from Cluster_extraction.KHM import OptimizedKHM
from sklearn.cluster import KMeans
import os


class ColorFeatureExtractor:
    """
    Color feature extractor based on two-stage clustering.

    Method:
    1. Stage 1: run KHM clustering independently on each image to obtain N
       color centers and their proportions within the image
    2. Stage 2: merge all color centers and perform global KMeans++ clustering
    3. Weight transfer: compute the representative weight of each final color
       from the stage-1 color center proportions
    4. Sorted output: colors sorted by weight in descending order

    Args:
        n_colors: number of colors to extract (default 4)
        use_two_stage: whether to use two-stage clustering (default: auto-detect)
        show_progress: whether to print progress information (default True)
    """

    def __init__(self, n_colors=4, use_two_stage=None, show_progress=True):
        self.n_colors = n_colors
        self.use_two_stage = use_two_stage
        self.show_progress = show_progress

    def extract_colors(self, image_paths):
        """
        Main function: extract dominant colors from images.

        Args:
            image_paths: a string or a list of strings; image path(s)

        Returns:
            For a single image: a list whose elements are (RGB color, weight) tuples
            For multiple images: a list of per-image color lists, each element
                being an (RGB color, weight) tuple
            Format: [(RGB color, weight), ...] or [[(RGB color, weight), ...], ...]
            Weight range: 0-1, summing to 1 (single image) or representative
                weights (multi-image fusion)
        """
        # Ensure the input is a list
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        # Automatically decide whether to use two-stage clustering
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
        Image reading function that supports paths containing Chinese characters.

        Method:
        1. Decode from binary data using cv2.imdecode
        2. Try an encoding conversion (GBK to UTF-8)
        3. Read directly (last resort)
        """
        # Method 1: read with cv2.imdecode
        try:
            with open(img_path, 'rb') as f:
                img_data = np.frombuffer(f.read(), dtype=np.uint8)
                img_bgr = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                if img_bgr is not None:
                    return img_bgr
        except Exception:
            pass

        # Method 2: if the file exists but reading failed, try an encoding conversion
        if os.path.exists(img_path):
            try:
                encoded_path = img_path.encode('gbk').decode('utf-8', errors='ignore')
                img_bgr = cv2.imread(encoded_path)
                if img_bgr is not None:
                    return img_bgr
            except:
                pass

            # Method 3: read directly
            img_bgr = cv2.imread(img_path)
            if img_bgr is not None:
                return img_bgr

        return None

    def _extract_colors_single_stage_with_proportion(self, image_paths):
        """
        Single-stage clustering: extract dominant colors independently from each
        image, sorted by proportion.

        Algorithm:
        1. Run KHM clustering on each image to obtain N color centers
        2. Compute the proportion of each color center within the image
        3. Sort colors by proportion in descending order

        Returns:
            A list of per-image color-weight lists, formatted as
            [[(RGB color, weight), ...], ...]
        """
        all_extracted_colors = []
        total_images = len(image_paths)

        for idx, img_path in enumerate(image_paths):
            if self.show_progress:
                print(f"处理图像 {idx + 1}/{total_images}: {os.path.basename(img_path)}")

            # Read the image
            img_bgr = self._read_image_with_chinese_path(img_path)
            if img_bgr is None:
                print(f"警告: 无法读取图像 {img_path}")
                continue

            # Convert to Lab color space
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)

            # Pixel sampling
            pixels = img_lab.reshape(-1, 3).astype(np.float32)

            # KHM clustering
            if self.show_progress:
                print("  正在进行KHM聚类...")

            khm = OptimizedKHM(n_clusters=self.n_colors)
            khm.fit(pixels)
            lab_colors = khm.cluster_centers_

            # Compute color proportions, sort, and return a weighted color list
            color_weight_pairs = self._calculate_and_sort_colors_with_weights(lab_colors, pixels)
            all_extracted_colors.append(color_weight_pairs)

            if self.show_progress:
                print(f"  完成，提取到 {len(color_weight_pairs)} 种主颜色")

        return all_extracted_colors

    def _extract_colors_two_stage_with_weight_transfer(self, image_paths):
        """
        Two-stage clustering: color extraction based on weight transfer.

        Algorithm:
        Stage 1:
        1. Run KHM clustering on each image to obtain N color centers
        2. Compute the proportion of each color center within the image

        Stage 2:
        1. Merge all color centers and perform global KMeans++ clustering
        2. Compute the representative weight of each final color via weight transfer
        3. Sort colors by weight in descending order

        Returns:
            A list of color-weight lists, formatted as [[(RGB color, weight), ...]]
        """
        if self.show_progress:
            print("=" * 60)
            print("第一阶段: 单图像KHM聚类与权重计算")
            print("=" * 60)

        # Stage-1 result storage
        stage1_results = []  # stores (color center, weight) pairs

        total_images = len(image_paths)

        for idx, img_path in enumerate(image_paths):
            if self.show_progress:
                print(f"\n处理图像 {idx + 1}/{total_images}: {os.path.basename(img_path)}")

            # Read the image
            img_bgr = self._read_image_with_chinese_path(img_path)
            if img_bgr is None:
                print(f"警告: 无法读取图像 {img_path}")
                continue

            # Convert to Lab color space
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)

            # Pixel sampling
            pixels = img_lab.reshape(-1, 3).astype(np.float32)

            # KHM clustering
            if self.show_progress:
                print("  正在进行KHM聚类...")

            khm = OptimizedKHM(n_clusters=self.n_colors)
            khm.fit(pixels)
            lab_colors = khm.cluster_centers_

            # Compute the proportion of each color center
            proportions = self._calculate_color_proportions(lab_colors, pixels)

            # Store stage-1 results
            for lab_color, proportion in zip(lab_colors, proportions):
                # Normalized weight: weights of each image sum to 1
                normalized_weight = proportion / total_images
                stage1_results.append({
                    'lab_color': lab_color,
                    'weight': normalized_weight,
                    'image_idx': idx
                })

            if self.show_progress:
                print(f"  完成，得到 {len(lab_colors)} 个颜色中心")
                print(f"  颜色中心权重: {[f'{p:.4f}' for p in proportions]}")

        if not stage1_results:
            print("错误: 没有成功读取任何图像")
            return [[[]]]

        if self.show_progress:
            print("\n" + "=" * 60)
            print(f"第一阶段完成，共收集 {len(stage1_results)} 个颜色中心")
            print("=" * 60)

            print("\n第二阶段: 全局KMeans++聚类与权重传递")
            print("=" * 60)

        # Extract all color centers
        all_lab_colors = np.array([result['lab_color'] for result in stage1_results])

        # Global KMeans++ clustering
        if self.show_progress:
            print(f"对 {len(all_lab_colors)} 个颜色中心进行KMeans++聚类...")

        kmeans = KMeans(n_clusters=self.n_colors, init='k-means++',
                        n_init=10, random_state=42)
        kmeans.fit(all_lab_colors)
        final_lab_colors = kmeans.cluster_centers_

        if self.show_progress:
            print("聚类完成，计算代表性权重...")

        # Weight transfer: assign each stage-1 color center to the nearest final color
        final_weights = np.zeros(self.n_colors)

        for result in stage1_results:
            lab_color = result['lab_color']
            weight = result['weight']

            # Compute distances to all final colors
            distances = np.linalg.norm(final_lab_colors - lab_color, axis=1)
            # Find the index of the nearest final color
            nearest_idx = np.argmin(distances)
            # Accumulate the weight
            final_weights[nearest_idx] += weight

        # Sort by weight in descending order
        sorted_indices = np.argsort(-final_weights)
        sorted_lab_colors = final_lab_colors[sorted_indices]
        sorted_weights = final_weights[sorted_indices]

        # Convert to RGB colors
        sorted_rgb_colors = self._lab_to_rgb_tuples(sorted_lab_colors)

        # Create the list of color-weight pairs
        color_weight_pairs = [(color, weight) for color, weight in zip(sorted_rgb_colors, sorted_weights)]

        if self.show_progress:
            print("\n" + "=" * 60)
            print("颜色提取完成")
            print("=" * 60)
            print(f"提取到 {len(color_weight_pairs)} 种主颜色:")
            for i, (color, weight) in enumerate(color_weight_pairs):
                print(f"  颜色 {i + 1}: RGB{color}, 代表性权重: {weight:.4f}")
            print("=" * 60)

        return [color_weight_pairs]

    def _calculate_color_proportions(self, lab_colors, pixels):
        """
        Compute the proportion of each color center within the pixel set.

        Args:
            lab_colors: Lab color array with shape (n_colors, 3)
            pixels: pixel array with shape (n_pixels, 3)

        Returns:
            Array of proportions for each color center, shape (n_colors,)
        """
        # Compute the distance from each pixel to each color center
        distances = np.sqrt(((pixels[:, np.newaxis, :] - lab_colors) ** 2).sum(axis=2))

        # Find the nearest color center for each pixel
        nearest_centers = np.argmin(distances, axis=1)

        # Count the pixels assigned to each color center
        center_counts = np.bincount(nearest_centers, minlength=len(lab_colors))

        # Compute proportions
        proportions = center_counts / len(pixels)

        return proportions

    def _calculate_and_sort_colors_with_weights(self, lab_colors, pixels):
        """
        Compute color proportions, sort by proportion in descending order, and
        return a color list that includes weights.

        Args:
            lab_colors: Lab color array with shape (n_colors, 3)
            pixels: pixel array with shape (n_pixels, 3)

        Returns:
            List of (RGB color, weight) tuples sorted by proportion
        """
        # Compute proportions
        proportions = self._calculate_color_proportions(lab_colors, pixels)

        # Sort by proportion in descending order
        sorted_indices = np.argsort(-proportions)
        sorted_lab_colors = lab_colors[sorted_indices]
        sorted_proportions = proportions[sorted_indices]

        # Convert to RGB
        sorted_rgb_colors = self._lab_to_rgb_tuples(sorted_lab_colors)

        # Create the list of color-weight pairs
        color_weight_pairs = [(color, weight) for color, weight in zip(sorted_rgb_colors, sorted_proportions)]

        return color_weight_pairs

    def _lab_to_rgb_tuples(self, lab_colors):
        """
        Convert a Lab color array to a list of RGB tuples.

        Args:
            lab_colors: Lab color array with shape (n_colors, 3)

        Returns:
            List of RGB color tuples: [(R,G,B), (R,G,B), ...]
        """
        rgb_tuples = []

        for lab_color in lab_colors:
            # Reshape into a 1x1 image for conversion
            lab_array = lab_color.reshape(1, 1, 3).astype(np.float32)

            # Ensure Lab values are within the correct ranges
            # OpenCV expects Lab ranges: L[0,100], a[-128,127], b[-128,127]
            # If the data is in the 0-255 range, it must be converted
            if np.max(lab_array) > 100:
                # Convert to the standard Lab ranges
                lab_array[..., 0] = lab_array[..., 0] * 100 / 255  # L channel
                lab_array[..., 1] = lab_array[..., 1] - 128  # a channel
                lab_array[..., 2] = lab_array[..., 2] - 128  # b channel

            # Lab to RGB
            rgb_array = cv2.cvtColor(lab_array, cv2.COLOR_LAB2RGB)
            # Get RGB values (0-1 range)
            rgb_color = rgb_array[0, 0]
            # Convert to the 0-255 range and clamp to valid values
            rgb_255 = np.clip(rgb_color * 255, 0, 255).astype(int)
            # Convert to a tuple and append to the list
            rgb_tuples.append(tuple(rgb_255))

        return rgb_tuples

    def _format_colors_for_output(self, colors_list):
        """
        Format color output (including weight information).

        Args:
            colors_list: a list of color lists, each element being
                [(RGB color, weight), ...]

        Returns:
            The formatted string
        """
        output = []

        if len(colors_list) == 1 and len(colors_list[0]) == self.n_colors:
            # Fused color output
            output.append("融合主颜色 (按代表性权重排序):")
            for i, (color, weight) in enumerate(colors_list[0]):
                output.append(f"  颜色 {i + 1}: RGB{color}, 权重: {weight:.4f}")
        else:
            # Independent color output
            output.append(f"独立提取的颜色 (共 {len(colors_list)} 张图像):")
            for i, color_weights in enumerate(colors_list):
                output.append(f"  图像 {i + 1}:")
                for j, (color, weight) in enumerate(color_weights):
                    output.append(f"    颜色 {j + 1}: RGB{color}, 占比: {weight:.4f}")

        return "\n".join(output)


# Usage example
if __name__ == "__main__":
    # Create a color extractor instance
    extractor = ColorFeatureExtractor(
        n_colors=4,  # extract 4 colors
        show_progress=True  # show progress information
    )
    #
    # # Example 1: single-image extraction
    # print("=" * 60)
    # print("Example 1: single-image color extraction")
    # print("=" * 60)
    #
    # single_image_path = "E:\ProjectX\Evaluation of Camo\image1.png"
    # single_colors = extractor.extract_colors(single_image_path)
    # print("Raw return value:", single_colors)
    # print("\nFormatted output:")
    # print(extractor._format_colors_for_output(single_colors))
    #
    # # Access individual colors and weights
    # print("\nAccess individual values:")
    # for i, (color, weight) in enumerate(single_colors[0]):
    #     print(f"Color {i + 1}: RGB{color}, weight: {weight:.4f}")

    # Example 2: multi-image fused extraction
    print("\n" + "=" * 60)
    print("Example 2: multi-image fused color extraction")
    print("=" * 60)

    multiple_image_paths = [
        r"E:\ProjectX\Test_data_env_fig\ukraine_P1.png",
        r"E:\ProjectX\Test_data_env_fig\ukraine_P2.png",
        r"E:\ProjectX\Test_data_env_fig\ukraine_P3.png"
    ]

    fused_colors = extractor.extract_colors(multiple_image_paths)
    print(extractor._format_colors_for_output(fused_colors))
    #
    # # Example 3: force single-stage clustering for multiple images
    # print("\n" + "=" * 60)
    # print("Example 3: independent color extraction for multiple images")
    # print("=" * 60)
    #
    # extractor_independent = ColorFeatureExtractor(
    #     n_colors=4,
    #     use_two_stage=False,  # force single-stage clustering
    #     show_progress=True
    # )
    #
    # independent_colors = extractor_independent.extract_colors(multiple_image_paths)
    # print(extractor_independent._format_colors_for_output(independent_colors))