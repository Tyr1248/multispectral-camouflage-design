"""
Visualization utilities
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from PIL import Image, ImageDraw, ImageFont
import random


def create_color_block_image(colors, block_size=100, arrangement='horizontal', spacing=10):
    """
    Create a color-block visualization image

    Args:
        colors: list[tuple] - list of colors
        block_size: int - size of each color block
        arrangement: str - arrangement mode
        spacing: int - spacing between color blocks

    Returns:
        np.array - color-block image
    """
    if not colors:
        return np.full((block_size, block_size, 3), 128, dtype=np.uint8)

    n_colors = len(colors)

    if arrangement == 'horizontal':
        width = n_colors * block_size + (n_colors - 1) * spacing
        height = block_size
    elif arrangement == 'vertical':
        width = block_size
        height = n_colors * block_size + (n_colors - 1) * spacing
    else:  # grid
        cols = int(np.ceil(np.sqrt(n_colors)))
        rows = int(np.ceil(n_colors / cols))
        width = cols * block_size + (cols - 1) * spacing
        height = rows * block_size + (rows - 1) * spacing

    # Create the image
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img.fill(240)

    # Draw the color blocks
    for i, color in enumerate(colors):
        if len(color) >= 3:
            r, g, b = color[:3]
        else:
            r, g, b = 128, 128, 128

        if arrangement == 'horizontal':
            x_start = i * (block_size + spacing)
            x_end = x_start + block_size
            y_start, y_end = 0, block_size
        elif arrangement == 'vertical':
            x_start, x_end = 0, block_size
            y_start = i * (block_size + spacing)
            y_end = y_start + block_size
        else:  # grid
            cols = int(np.ceil(np.sqrt(n_colors)))
            row = i // cols
            col = i % cols
            x_start = col * (block_size + spacing)
            x_end = x_start + block_size
            y_start = row * (block_size + spacing)
            y_end = y_start + block_size

        img[y_start:y_end, x_start:x_end] = [r, g, b]

        # Add a border
        border_color = [max(0, c - 40) for c in [r, g, b]]
        img[y_start, x_start:x_end] = border_color
        img[y_end-1, x_start:x_end] = border_color
        img[y_start:y_end, x_start] = border_color
        img[y_start:y_end, x_end-1] = border_color

    return img


def plot_spectrum_comparison(spectrum_data, reference_data=None, title="光谱特性比较"):
    """
    Plot a spectrum comparison chart

    Args:
        spectrum_data: dict - main spectrum data
        reference_data: dict - reference spectrum data (optional)
        title: str - chart title

    Returns:
        matplotlib.figure.Figure - the plotted figure
    """
    fig = Figure(figsize=(10, 6), dpi=100)
    ax = fig.add_subplot(111)

    # Plot the main spectrum
    if spectrum_data and 'wavelengths' in spectrum_data and 'reflectance' in spectrum_data:
        wavelengths = spectrum_data['wavelengths']
        reflectance = spectrum_data['reflectance']

        ax.plot(wavelengths, reflectance, 'b-', linewidth=2, label='设计光谱')

        # Mark key spectral regions
        ax.axvspan(380, 780, alpha=0.1, color='yellow', label='可见光')
        ax.axvspan(780, 1400, alpha=0.1, color='red', label='近红外')
        ax.axvspan(1400, 2500, alpha=0.1, color='darkred', label='短波红外')

        # Compute and display the average reflectance
        avg_reflectance = np.mean(reflectance)
        ax.axhline(y=avg_reflectance, color='b', linestyle='--', alpha=0.5,
                   label=f'平均反射率: {avg_reflectance:.3f}')

    # Plot the reference spectrum
    if reference_data and 'wavelengths' in reference_data and 'reflectance' in reference_data:
        ax.plot(reference_data['wavelengths'], reference_data['reflectance'],
                'r--', linewidth=1.5, label='参考光谱')

    # Set chart properties
    ax.set_xlabel('波长 (nm)', fontsize=12)
    ax.set_ylabel('反射率', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    # Set the axis ranges
    if spectrum_data and 'wavelengths' in spectrum_data:
        ax.set_xlim([spectrum_data['wavelengths'][0], spectrum_data['wavelengths'][-1]])
    ax.set_ylim([0, 1])

    fig.tight_layout()
    return fig


def generate_design_preview(design_params, pattern_image=None, size=(800, 600), include_params=True):
    """
    Generate a design preview image (amorphous/crystalline camouflage stacked for comparison)

    Args:
        design_params: dict - design parameters
        pattern_image: np.array or dict - camouflage pattern;
            pass {'amorphous': arr, 'crystalline': arr} to generate a dual-state comparison
        size: tuple - preview image size
        include_params: bool - whether to append parameter text at the bottom

    Returns:
        np.array - design preview image
    """
    width, height = size

    # Normalize the pattern input: accept a single array or an {'amorphous':..., 'crystalline':...} dict
    patterns = {}
    if isinstance(pattern_image, dict):
        patterns = {k: v for k, v in pattern_image.items() if isinstance(v, np.ndarray)}
    elif isinstance(pattern_image, np.ndarray):
        patterns = {'amorphous': pattern_image}

    img = Image.new('RGB', (width, height), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    ordered = [(k, lbl) for k, lbl in
               [('amorphous', 'Amorphous State'), ('crystalline', 'Crystalline State')]
               if k in patterns]

    if ordered:
        # Reserve space at the bottom for the parameter text
        footer_h = 46 if include_params else 0
        label_h = 30
        n = len(ordered)
        block_h = max(1, (height - footer_h) // n)

        y = 0
        for key, label in ordered:
            pattern_arr = patterns[key]
            pattern_img = Image.fromarray(pattern_arr.astype('uint8')).convert('RGB')

            # Scale into the region preserving the original aspect ratio (contain),
            # centered horizontally, without stretching distortion
            region_h = max(1, block_h - label_h)
            pw, ph = pattern_img.size
            scale = min(width / pw, region_h / ph)
            new_w = max(1, int(pw * scale))
            new_h = max(1, int(ph * scale))
            pattern_img = pattern_img.resize((new_w, new_h), Image.LANCZOS)
            x_offset = (width - new_w) // 2

            # Label bar
            draw.rectangle([0, y, width, y + label_h], fill=(52, 73, 94))
            draw.text((10, y + 7), label, fill=(255, 255, 255), font=font)
            img.paste(pattern_img, (x_offset, y + label_h))
            y += block_h

        # Parameter text at the bottom (in English, since the default font cannot render Chinese)
        if include_params:
            lines = []
            design_type = design_params.get('design_type', 'N/A')
            lines.append(f"Design type: {design_type}")
            design_time = design_params.get('design_time_seconds')
            if design_time is not None:
                lines.append(f"Design time: {design_time:.1f} s")
            for i, line in enumerate(lines):
                draw.text((10, height - footer_h + 6 + i * 18), line,
                          fill=(30, 30, 30), font=font)
    else:
        # Fallback when no pattern is given: draw color blocks
        colors = design_params.get('amorphous_colors', []) + design_params.get('crystalline_colors', [])
        if colors:
            color_block_width = width // min(len(colors), 5)
            for i, color in enumerate(colors[:5]):
                if len(color) >= 3:
                    r, g, b = int(color[0]), int(color[1]), int(color[2])
                else:
                    r, g, b = 128, 128, 128
                x_start = i * color_block_width
                x_end = (i + 1) * color_block_width
                draw.rectangle([x_start, 0, x_end, 100], fill=(r, g, b))

    return np.array(img)
