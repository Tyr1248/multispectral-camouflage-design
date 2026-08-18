import os
import numpy as np
from pathlib import Path
from PIL import Image
import cv2


class SpotDatabaseManager:
    """Spot database manager."""

    def __init__(self, spot_database_path='spot_database'):
        """
        Initialize the spot database manager.

        Args:
            spot_database_path: Path to the spot database.
        """
        self.spot_database_path = spot_database_path
        self.spot_database = self.load_spot_database()

    def load_spot_database(self):
        """
        Load the spot database.

        Returns:
            Spot database dictionary with 'large' and 'small' keys.
        """
        spot_db = {'large': [], 'small': []}

        # Load large spots
        large_path = Path(self.spot_database_path) / 'large_spots'
        if large_path.exists():
            large_files = list(large_path.glob('*.png'))
            for filepath in large_files:
                spot_data = self.load_spot_image(filepath)
                if spot_data is not None:
                    spot_db['large'].append(spot_data)

        # Load small spots
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
        Load a single spot image.

        Args:
            filepath: Path to the image file.

        Returns:
            Spot data dictionary with image, filename, size, etc.
        """
        try:
            # Read the image
            with Image.open(filepath) as img:
                image = np.array(img)

            # Convert to grayscale if the image is colored
            if len(image.shape) == 3:
                if image.shape[2] == 4:  # RGBA
                    # Split out the alpha channel
                    alpha = image[:, :, 3]
                    # Convert RGB to grayscale
                    gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2GRAY)
                    # Apply the alpha channel
                    gray = gray * (alpha / 255)
                else:  # RGB
                    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image

            # Binarize with threshold 127
            binary_spot = gray > 127

            # Extract size information from the filename
            filename = Path(filepath).stem
            size_str = filename.split('_')[0]  # e.g. "12x12"
            try:
                width, height = map(int, size_str.split('x'))
            except:
                # Fall back to the actual image size if parsing fails
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