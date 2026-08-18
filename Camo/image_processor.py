import numpy as np
import cv2


class ImageProcessor:
    """Image processor."""

    @staticmethod
    def crop_to_original_size(canvas, expand_pixels, original_size):
        """
        Crop the expanded canvas back to the original size.

        Args:
            canvas: Expanded canvas.
            expand_pixels: Number of expanded pixels.
            original_size: Original size (width, height).

        Returns:
            Cropped canvas.
        """
        if expand_pixels == 0:
            return canvas

        height, width = canvas.shape[:2]
        start_x = expand_pixels
        start_y = expand_pixels
        end_x = width - expand_pixels
        end_y = height - expand_pixels

        # Ensure indices are valid
        start_x = max(0, min(start_x, width - 1))
        start_y = max(0, min(start_y, height - 1))
        end_x = max(start_x + 1, min(end_x, width))
        end_y = max(start_y + 1, min(end_y, height))

        cropped_canvas = canvas[start_y:end_y, start_x:end_x, :]
        return cropped_canvas

    @staticmethod
    def upscale_image(image, scale_factor, method='nearest'):
        """
        Upsample an image.

        Args:
            image: Input image.
            scale_factor: Scaling factor.
            method: Interpolation method ('nearest', 'bilinear', 'cubic').

        Returns:
            Upsampled image.
        """
        if scale_factor <= 1:
            return image

        height, width = image.shape[:2]
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

        # Select interpolation method
        if method == 'nearest':
            interpolation = cv2.INTER_NEAREST
        elif method == 'bilinear':
            interpolation = cv2.INTER_LINEAR
        elif method == 'cubic':
            interpolation = cv2.INTER_CUBIC
        else:
            interpolation = cv2.INTER_NEAREST

        # Single-channel image
        if len(image.shape) == 2:
            upscaled_image = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
        else:
            # Multi-channel image
            upscaled_image = np.zeros((new_height, new_width, image.shape[2]), dtype=image.dtype)
            for c in range(image.shape[2]):
                upscaled_image[:, :, c] = cv2.resize(
                    image[:, :, c], (new_width, new_height), interpolation=interpolation
                )

        print(f"上采样完成: {width}x{height} -> {new_width}x{new_height} (倍率: {scale_factor}x)")
        return upscaled_image