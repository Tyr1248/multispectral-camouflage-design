import numpy as np
import cv2


class SpotRenderer:
    """斑点渲染器"""

    def rotate_spot(self, spot_data, angle):
        """
        旋转斑点

        Args:
            spot_data: 斑点数据字典
            angle: 旋转角度 (0, 90, 180, 270)

        Returns:
            旋转后的斑点数据
        """
        if angle == 0:
            return spot_data

        spot_image = spot_data['image']
        original_size = spot_data['original_size']

        # 根据角度旋转图像
        if angle == 90:
            rotated_image = np.rot90(spot_image, 1)
            new_size = (original_size[1], original_size[0])  # 交换宽高
        elif angle == 180:
            rotated_image = np.rot90(spot_image, 2)
            new_size = original_size
        elif angle == 270:
            rotated_image = np.rot90(spot_image, 3)
            new_size = (original_size[1], original_size[0])
        else:
            # 使用OpenCV进行任意角度旋转（慢但准确）
            height, width = spot_image.shape[:2]
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_image = cv2.warpAffine(spot_image.astype(np.float32), rotation_matrix, (width, height))
            rotated_image = rotated_image > 0.5  # 重新二值化
            new_size = original_size

        rotated_spot = {
            'image': rotated_image,
            'filename': spot_data['filename'],
            'size': new_size,
            'original_size': original_size
        }

        return rotated_spot

    def calculate_overlap_ratio(self, coverage_mask, x, y, spot_width, spot_height, spot_image):
        """
        计算重叠率

        Args:
            coverage_mask: 覆盖掩码 (bool数组)
            x, y: 斑点左上角位置
            spot_width, spot_height: 斑点尺寸
            spot_image: 斑点二值图像

        Returns:
            重叠比例 (0-1)
        """
        # 获取掩码的ROI区域
        roi_mask = coverage_mask[y:y + spot_height, x:x + spot_width]

        # 确保尺寸匹配
        if roi_mask.shape != spot_image.shape:
            # 调整掩码尺寸以匹配斑点
            min_height = min(roi_mask.shape[0], spot_image.shape[0])
            min_width = min(roi_mask.shape[1], spot_image.shape[1])
            roi_mask = roi_mask[:min_height, :min_width]
            spot_image_cropped = spot_image[:min_height, :min_width]
        else:
            spot_image_cropped = spot_image

        # 计算重叠像素数
        overlap_pixels = np.sum(roi_mask & spot_image_cropped)
        total_pixels = np.sum(spot_image_cropped)

        if total_pixels > 0:
            overlap_ratio = overlap_pixels / total_pixels
        else:
            overlap_ratio = 0.0

        return overlap_ratio