"""
可视化功能
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from PIL import Image, ImageDraw
import random


def create_color_block_image(colors, block_size=100, arrangement='horizontal', spacing=10):
    """
    创建颜色块可视化图像

    输入:
        colors: list[tuple] - 颜色列表
        block_size: int - 每个颜色块的大小
        arrangement: str - 排列方式
        spacing: int - 颜色块间距

    输出:
        np.array - 颜色块图像
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

    # 创建图像
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img.fill(240)

    # 绘制颜色块
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

        # 添加边框
        border_color = [max(0, c - 40) for c in [r, g, b]]
        img[y_start, x_start:x_end] = border_color
        img[y_end-1, x_start:x_end] = border_color
        img[y_start:y_end, x_start] = border_color
        img[y_start:y_end, x_end-1] = border_color

    return img


def plot_spectrum_comparison(spectrum_data, reference_data=None, title="光谱特性比较"):
    """
    绘制光谱比较图

    输入:
        spectrum_data: dict - 主要光谱数据
        reference_data: dict - 参考光谱数据（可选）
        title: str - 图表标题

    输出:
        matplotlib.figure.Figure - 绘制好的图表
    """
    fig = Figure(figsize=(10, 6), dpi=100)
    ax = fig.add_subplot(111)

    # 绘制主要光谱
    if spectrum_data and 'wavelengths' in spectrum_data and 'reflectance' in spectrum_data:
        wavelengths = spectrum_data['wavelengths']
        reflectance = spectrum_data['reflectance']

        ax.plot(wavelengths, reflectance, 'b-', linewidth=2, label='设计光谱')

        # 标记关键区域
        ax.axvspan(380, 780, alpha=0.1, color='yellow', label='可见光')
        ax.axvspan(780, 1400, alpha=0.1, color='red', label='近红外')
        ax.axvspan(1400, 2500, alpha=0.1, color='darkred', label='短波红外')

        # 计算并显示平均反射率
        avg_reflectance = np.mean(reflectance)
        ax.axhline(y=avg_reflectance, color='b', linestyle='--', alpha=0.5,
                   label=f'平均反射率: {avg_reflectance:.3f}')

    # 绘制参考光谱
    if reference_data and 'wavelengths' in reference_data and 'reflectance' in reference_data:
        ax.plot(reference_data['wavelengths'], reference_data['reflectance'],
                'r--', linewidth=1.5, label='参考光谱')

    # 设置图表属性
    ax.set_xlabel('波长 (nm)', fontsize=12)
    ax.set_ylabel('反射率', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    # 设置坐标轴范围
    if spectrum_data and 'wavelengths' in spectrum_data:
        ax.set_xlim([spectrum_data['wavelengths'][0], spectrum_data['wavelengths'][-1]])
    ax.set_ylim([0, 1])

    fig.tight_layout()
    return fig


def generate_design_preview(design_params, pattern_image=None, size=(400, 300), include_params=True):
    """
    生成设计预览图

    输入:
        design_params: dict - 设计参数
        pattern_image: np.array - 迷彩图案（可选）
        size: tuple - 预览图尺寸
        include_params: bool - 是否包含参数文本

    输出:
        np.array - 设计预览图像
    """
    width, height = size

    # 创建PIL图像
    img = Image.new('RGB', (width, height), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)

    # 如果有图案图像，将其添加到预览中
    if pattern_image is not None and isinstance(pattern_image, np.ndarray):
        pattern_img = Image.fromarray(pattern_image)
        pattern_img = pattern_img.resize((width, int(height * 0.6)))
        img.paste(pattern_img, (0, 0))
        text_y_start = pattern_img.height + 10
    else:
        # 绘制颜色块
        colors = design_params.get('amorphous_colors', []) + design_params.get('crystalline_colors', [])
        if colors:
            color_block_width = width // min(len(colors), 5)
            for i, color in enumerate(colors[:5]):
                if len(color) >= 3:
                    r, g, b = color[:3]
                else:
                    r, g, b = 128, 128, 128

                x_start = i * color_block_width
                x_end = (i + 1) * color_block_width
                draw.rectangle([x_start, 0, x_end, 100], fill=(r, g, b))

        text_y_start = 110

    # 如果包含参数，添加文本信息
    if include_params and text_y_start < height - 50:
        try:
            font = ImageFont.load_default()

            lines = []

            # 设计类型
            design_type = design_params.get('design_type', '未知')
            lines.append(f"设计类型: {design_type}")

            # 厚度和折射率
            thickness = design_params.get('thickness', 'N/A')
            refractive_index = design_params.get('refractive_index', 'N/A')
            lines.append(f"厚度: {thickness}nm, 折射率: {refractive_index}")

            # 颜色组信息
            amorphous_count = len(design_params.get('amorphous_colors', []))
            crystalline_count = len(design_params.get('crystalline_colors', []))
            if amorphous_count or crystalline_count:
                lines.append(f"颜色组: {amorphous_count}非晶态 + {crystalline_count}晶态")

            # 绘制文本
            for i, line in enumerate(lines):
                y_pos = text_y_start + i * 20
                if y_pos < height - 20:
                    draw.text((10, y_pos), line, fill=(0, 0, 0), font=font)

        except Exception as e:
            print(f"生成预览文本时出错: {e}")

    # 转换为numpy数组
    return np.array(img)