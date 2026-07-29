import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


def plot_gst_color_comparison(Lab_crystalline, Lab_amorphous, deltaE, thicknesses=None, save_path=None):
    """
    绘制GST晶态和非晶态颜色对比图

    参数:
    Lab_crystalline: 晶态GST的Lab颜色值
    Lab_amorphous: 非晶态GST的Lab颜色值
    deltaE: 颜色差异ΔE
    thicknesses: 厚度数组（可选，用于标题）
    save_path: 图片保存路径（可选）
    """

    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # 子图1: Lab颜色空间可视化
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('a*')
    ax1.set_ylabel('b*')
    ax1.set_title('Lab Color Space - GST States')

    # 绘制Lab坐标点
    ax1.scatter(Lab_crystalline[1], Lab_crystalline[2], c='blue', s=200,
                label=f'Crystalline (L*={Lab_crystalline[0]:.1f})', edgecolors='black')
    ax1.scatter(Lab_amorphous[1], Lab_amorphous[2], c='red', s=200,
                label=f'Amorphous (L*={Lab_amorphous[0]:.1f})', edgecolors='black')

    # 连接两点显示ΔE
    ax1.plot([Lab_crystalline[1], Lab_amorphous[1]],
             [Lab_crystalline[2], Lab_amorphous[2]], 'k--', alpha=0.5)
    ax1.text((Lab_crystalline[1] + Lab_amorphous[1]) / 2,
             (Lab_crystalline[2] + Lab_amorphous[2]) / 2,
             f'ΔE = {deltaE:.2f}', ha='center', va='bottom',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax1.legend()

    # 子图2: 颜色块对比
    ax2.axis('off')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_title('Color Appearance Comparison')

    # 将Lab转换为RGB用于显示
    from colour import Lab_to_XYZ, XYZ_to_sRGB
    import numpy as np

    # 转换晶态Lab到RGB
    XYZ_crystalline = Lab_to_XYZ(Lab_crystalline)
    RGB_crystalline = XYZ_to_sRGB(XYZ_crystalline)
    RGB_crystalline = np.clip(RGB_crystalline, 0, 1)

    # 转换非晶态Lab到RGB
    XYZ_amorphous = Lab_to_XYZ(Lab_amorphous)
    RGB_amorphous = XYZ_to_sRGB(XYZ_amorphous)
    RGB_amorphous = np.clip(RGB_amorphous, 0, 1)

    # 添加颜色块
    rect_height = 0.3
    rect_width = 0.3
    spacing = 0.1

    # 晶态颜色块
    ax2.add_patch(Rectangle((0.2, 0.6), rect_width, rect_height,
                            facecolor=RGB_crystalline, edgecolor='black', linewidth=3))
    ax2.text(0.2 + rect_width / 2, 0.6 - 0.05, 'Crystalline GST',
             ha='center', va='top', fontsize=12, weight='bold')

    # 非晶态颜色块
    ax2.add_patch(Rectangle((0.6, 0.6), rect_width, rect_height,
                            facecolor=RGB_amorphous, edgecolor='black', linewidth=3))
    ax2.text(0.6 + rect_width / 2, 0.6 - 0.05, 'Amorphous GST',
             ha='center', va='top', fontsize=12, weight='bold')

    # 添加详细的颜色信息
    info_y = 0.4
    line_height = 0.05

    # 晶态信息
    ax2.text(0.35, info_y, 'Crystalline State:', ha='center', va='top',
             fontsize=11, weight='bold', color='blue')
    ax2.text(0.35, info_y - line_height, f'L* = {Lab_crystalline[0]:.2f}',
             ha='center', va='top', fontsize=10)
    ax2.text(0.35, info_y - 2 * line_height, f'a* = {Lab_crystalline[1]:.2f}',
             ha='center', va='top', fontsize=10)
    ax2.text(0.35, info_y - 3 * line_height, f'b* = {Lab_crystalline[2]:.2f}',
             ha='center', va='top', fontsize=10)

    # 非晶态信息
    ax2.text(0.75, info_y, 'Amorphous State:', ha='center', va='top',
             fontsize=11, weight='bold', color='red')
    ax2.text(0.75, info_y - line_height, f'L* = {Lab_amorphous[0]:.2f}',
             ha='center', va='top', fontsize=10)
    ax2.text(0.75, info_y - 2 * line_height, f'a* = {Lab_amorphous[1]:.2f}',
             ha='center', va='top', fontsize=10)
    ax2.text(0.75, info_y - 3 * line_height, f'b* = {Lab_amorphous[2]:.2f}',
             ha='center', va='top', fontsize=10)

    # 添加ΔE信息
    ax2.text(0.5, 0.2, f'Color Difference: ΔE = {deltaE:.2f}',
             ha='center', va='top', fontsize=14, weight='bold',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

    # 如果有厚度信息，添加到标题
    title = "GST Colorimetric Analysis"
    if thicknesses is not None:
        thickness_str = ", ".join([f"{t:.1f}nm" for t in thicknesses])
        title += f"\nThicknesses: [{thickness_str}]"

    # 调整标题位置，向上移动一点
    plt.suptitle(title, fontsize=12, weight='bold', y=0.98)  # 将y从默认的0.98调整为0.98
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)  # 增加顶部空间，使标题更靠上

    # 保存图片（如果指定了路径）
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存至: {save_path}")

    plt.show()


# 使用示例
if __name__ == "__main__":
    # 示例Lab值（从计算函数获得）
    Lab_crystalline = [65.32, 12.45, -8.76]  # 示例值
    Lab_amorphous = [45.67, -5.43, 15.21]  # 示例值
    deltaE = 28.45  # 示例ΔE值
    thicknesses = [121, 4, 85, 39, 800]  # 示例厚度

    # 绘制对比图
    plot_gst_color_comparison(Lab_crystalline, Lab_amorphous, deltaE, thicknesses)