import numpy as np
import cv2


class SpotRenderer:
    """Spot renderer."""

    def rotate_spot(self, spot_data, angle):
        """
        Rotate a spot.

        Args:
            spot_data: Spot data dictionary.
            angle: Rotation angle (0, 90, 180, 270).

        Returns:
            Rotated spot data.
        """
        if angle == 0:
            return spot_data

        spot_image = spot_data['image']
        original_size = spot_data['original_size']

        # Rotate the image according to the angle
        if angle == 90:
            rotated_image = np.rot90(spot_image, 1)
            new_size = (original_size[1], original_size[0])  # swap width and height
        elif angle == 180:
            rotated_image = np.rot90(spot_image, 2)
            new_size = original_size
        elif angle == 270:
            rotated_image = np.rot90(spot_image, 3)
            new_size = (original_size[1], original_size[0])
        else:
            # Arbitrary-angle rotation with OpenCV (slow but accurate)
            height, width = spot_image.shape[:2]
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_image = cv2.warpAffine(spot_image.astype(np.float32), rotation_matrix, (width, height))
            rotated_image = rotated_image > 0.5  # re-binarize
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
        Calculate the overlap ratio.

        Args:
            coverage_mask: Coverage mask (bool array).
            x, y: Top-left position of the spot.
            spot_width, spot_height: Spot dimensions.
            spot_image: Binary spot image.

        Returns:
            Overlap ratio (0-1).
        """
        # Get the ROI of the mask
        roi_mask = coverage_mask[y:y + spot_height, x:x + spot_width]

        # Ensure sizes match
        if roi_mask.shape != spot_image.shape:
            # Crop the mask to match the spot
            min_height = min(roi_mask.shape[0], spot_image.shape[0])
            min_width = min(roi_mask.shape[1], spot_image.shape[1])
            roi_mask = roi_mask[:min_height, :min_width]
            spot_image_cropped = spot_image[:min_height, :min_width]
        else:
            spot_image_cropped = spot_image

        # Count overlapping pixels
        overlap_pixels = np.sum(roi_mask & spot_image_cropped)
        total_pixels = np.sum(spot_image_cropped)

        if total_pixels > 0:
            overlap_ratio = overlap_pixels / total_pixels
        else:
            overlap_ratio = 0.0

        return overlap_ratio