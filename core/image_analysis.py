"""
Image analysis utilities
"""

import random
import numpy as np
from PIL import Image
import cv2


def load_and_preprocess_image(image_path, resize_to=(800, 600)):
    """
    Load and preprocess an image

    Args:
        image_path: str - image path
        resize_to: tuple - target size (optional)

    Returns:
        np.array - preprocessed image array (RGB format)
    """
    try:
        # Load the image with PIL
        img = Image.open(image_path)

        # Convert to RGB (if RGBA)
        if img.mode in ('RGBA', 'LA'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[-1])
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')

        # Resize
        if resize_to:
            img = img.resize(resize_to, Image.Resampling.LANCZOS)

        # Convert to a numpy array
        img_array = np.array(img)

        return img_array

    except Exception as e:
        print(f"图像加载失败: {e}")
        # Return a random test image as fallback
        return np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)


def analyze_environment_texture(environment_image):
    """
    Analyze environment texture features

    Args:
        environment_image: np.array - environment image

    Returns:
        dict - texture feature dictionary
    """
    # Mock implementation: return random texture features
    return {
        'color_palette': [
            (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            for _ in range(3)
        ],
        'texture_density': random.uniform(0.2, 0.8),
        'contrast': random.uniform(0.3, 0.9),
        'complexity': random.uniform(0.4, 0.95),
        'dominant_orientation': random.uniform(0, 180),
        'roughness': random.uniform(0.1, 0.9),
    }


def calculate_color_statistics(image, mask=None):
    """
    Compute color statistics of an image

    Args:
        image: np.array - image
        mask: np.array - mask (optional)

    Returns:
        dict - statistics dictionary
    """
    if image is None:
        return {}

    # Apply the mask if provided
    if mask is not None and mask.shape[:2] == image.shape[:2]:
        masked_image = image[mask > 0]
        if len(masked_image) == 0:
            pixels = image.reshape(-1, 3)
        else:
            pixels = masked_image
    else:
        pixels = image.reshape(-1, 3)

    # Compute statistics
    mean = np.mean(pixels, axis=0)
    std = np.std(pixels, axis=0)

    # Compute color histograms
    hist_red = np.histogram(pixels[:, 0], bins=32, range=(0, 255))[0]
    hist_green = np.histogram(pixels[:, 1], bins=32, range=(0, 255))[0]
    hist_blue = np.histogram(pixels[:, 2], bins=32, range=(0, 255))[0]

    # Find the most frequent color as the dominant color
    hist_3d, _ = np.histogramdd(pixels, bins=(8, 8, 8), range=[(0, 255), (0, 255), (0, 255)])
    max_idx = np.unravel_index(np.argmax(hist_3d), hist_3d.shape)
    dominant_color = (
        int((max_idx[0] + 0.5) * 255 / 8),
        int((max_idx[1] + 0.5) * 255 / 8),
        int((max_idx[2] + 0.5) * 255 / 8)
    )

    return {
        'mean': tuple(mean.astype(int)),
        'std': tuple(std.astype(int)),
        'dominant_color': dominant_color,
        'histogram': {
            'red': hist_red.tolist(),
            'green': hist_green.tolist(),
            'blue': hist_blue.tolist(),
        },
        'pixel_count': len(pixels),
    }
