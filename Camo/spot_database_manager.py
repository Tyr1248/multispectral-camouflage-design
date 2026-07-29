import os
import numpy as np
from pathlib import Path
from PIL import Image
import cv2


class SpotDatabaseManager:
    """斑点数据库管理器"""

    def __init__(self, spot_database_path='spot_database'):
        """
        初始化斑点数据库管理器

        Args:
            spot_database_path: 斑点数据库路径
        """
        self.spot_database_path = spot_database_path
        self.spot_database = self.load_spot_database()

    def load_spot_database(self):
        """
        加载斑点数据库

        Returns:
            斑点数据库字典，包含'large'和'small'键
        """
        spot_db = {'large': [], 'small': []}

        # 加载大斑点
        large_path = Path(self.spot_database_path) / 'large_spots'
        if large_path.exists():
            large_files = list(large_path.glob('*.png'))
            for filepath in large_files:
                spot_data = self.load_spot_image(filepath)
                if spot_data is not None:
                    spot_db['large'].append(spot_data)

        # 加载小斑点
        small_path = Path(self.spot_database_path) / 'small_spots'
        if small_path.exists():
            small_files = list(small_path.glob('*.png'))
            for filepath in small_files:
                spot_data = self.load_spot_image(filepath)
                if spot_data is not None:
                    spot_db['small'].append(spot_data)

        print(f"斑点数据库加载完成: {len(spot_db['large'])} 个大斑点, {len(spot_db['small'])} 个小斑点")
        return spot_db

    def load_spot_image(self, filepath):
        """
        加载单个斑点图像

        Args:
            filepath: 图像文件路径

        Returns:
            斑点数据字典，包含图像、文件名、尺寸等信息
        """
        try:
            # 读取图像
            with Image.open(filepath) as img:
                image = np.array(img)

            # 如果图像是彩色，转换为灰度
            if len(image.shape) == 3:
                if image.shape[2] == 4:  # RGBA
                    # 分离alpha通道
                    alpha = image[:, :, 3]
                    # 将RGB转换为灰度
                    gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2GRAY)
                    # 应用alpha通道
                    gray = gray * (alpha / 255)
                else:  # RGB
                    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image

            # 二值化 - 阈值设置为127
            binary_spot = gray > 127

            # 从文件名提取尺寸信息
            filename = Path(filepath).stem
            size_str = filename.split('_')[0]  # 例如 "12x12"
            try:
                width, height = map(int, size_str.split('x'))
            except:
                # 如果无法解析尺寸，使用图像实际尺寸
                height, width = binary_spot.shape
                print(f"警告: 无法从文件名解析尺寸，使用图像实际尺寸 {height}x{width}")

            spot_data = {
                'image': binary_spot,
                'filename': filename,
                'size': (width, height),
                'original_size': (width, height)
            }

            return spot_data

        except Exception as e:
            print(f"加载斑点图像时出错 {filepath}: {e}")
            return None