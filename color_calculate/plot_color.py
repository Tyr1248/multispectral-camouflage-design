import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


def plot_gst_color_comparison(Lab_crystalline, Lab_amorphous, deltaE, thicknesses=None, save_path=None):
    """
    Plot a color comparison between crystalline and amorphous GST.

    Args:
        Lab_crystalline: Lab color values of crystalline GST.
        Lab_amorphous: Lab color values of amorphous GST.
        deltaE: color difference ΔE.
        thicknesses: thickness array (optional, used in the title).
        save_path: path to save the figure (optional).
    """

    # Create the figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Subplot 1: Lab color space visualization
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('a*')
    ax1.set_ylabel('b*')
    ax1.set_title('Lab Color Space - GST States')

    # Plot the Lab coordinate points
    ax1.scatter(Lab_crystalline[1], Lab_crystalline[2], c='blue', s=200,
                label=f'Crystalline (L*={Lab_crystalline[0]:.1f})', edgecolors='black')
    ax1.scatter(Lab_amorphous[1], Lab_amorphous[2], c='red', s=200,
                label=f'Amorphous (L*={Lab_amorphous[0]:.1f})', edgecolors='black')

    # Connect the two points to show ΔE
    ax1.plot([Lab_crystalline[1], Lab_amorphous[1]],
             [Lab_crystalline[2], Lab_amorphous[2]], 'k--', alpha=0.5)
    ax1.text((Lab_crystalline[1] + Lab_amorphous[1]) / 2,
             (Lab_crystalline[2] + Lab_amorphous[2]) / 2,
             f'ΔE = {deltaE:.2f}', ha='center', va='bottom',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax1.legend()

    # Subplot 2: color swatch comparison
    ax2.axis('off')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_title('Color Appearance Comparison')

    # Convert Lab to RGB for display
    from colour import Lab_to_XYZ, XYZ_to_sRGB
    import numpy as np

    # Convert crystalline Lab to RGB
    XYZ_crystalline = Lab_to_XYZ(Lab_crystalline)
    RGB_crystalline = XYZ_to_sRGB(XYZ_crystalline)
    RGB_crystalline = np.clip(RGB_crystalline, 0, 1)

    # Convert amorphous Lab to RGB
    XYZ_amorphous = Lab_to_XYZ(Lab_amorphous)
    RGB_amorphous = XYZ_to_sRGB(XYZ_amorphous)
    RGB_amorphous = np.clip(RGB_amorphous, 0, 1)

    # Add the color swatches
    rect_height = 0.3
    rect_width = 0.3
    spacing = 0.1

    # Crystalline color swatch
    ax2.add_patch(Rectangle((0.2, 0.6), rect_width, rect_height,
                            facecolor=RGB_crystalline, edgecolor='black', linewidth=3))
    ax2.text(0.2 + rect_width / 2, 0.6 - 0.05, 'Crystalline GST',
             ha='center', va='top', fontsize=12, weight='bold')

    # Amorphous color swatch
    ax2.add_patch(Rectangle((0.6, 0.6), rect_width, rect_height,
                            facecolor=RGB_amorphous, edgecolor='black', linewidth=3))
    ax2.text(0.6 + rect_width / 2, 0.6 - 0.05, 'Amorphous GST',
             ha='center', va='top', fontsize=12, weight='bold')

    # Add detailed color information
    info_y = 0.4
    line_height = 0.05

    # Crystalline state info
    ax2.text(0.35, info_y, 'Crystalline State:', ha='center', va='top',
             fontsize=11, weight='bold', color='blue')
    ax2.text(0.35, info_y - line_height, f'L* = {Lab_crystalline[0]:.2f}',
             ha='center', va='top', fontsize=10)
    ax2.text(0.35, info_y - 2 * line_height, f'a* = {Lab_crystalline[1]:.2f}',
             ha='center', va='top', fontsize=10)
    ax2.text(0.35, info_y - 3 * line_height, f'b* = {Lab_crystalline[2]:.2f}',
             ha='center', va='top', fontsize=10)

    # Amorphous state info
    ax2.text(0.75, info_y, 'Amorphous State:', ha='center', va='top',
             fontsize=11, weight='bold', color='red')
    ax2.text(0.75, info_y - line_height, f'L* = {Lab_amorphous[0]:.2f}',
             ha='center', va='top', fontsize=10)
    ax2.text(0.75, info_y - 2 * line_height, f'a* = {Lab_amorphous[1]:.2f}',
             ha='center', va='top', fontsize=10)
    ax2.text(0.75, info_y - 3 * line_height, f'b* = {Lab_amorphous[2]:.2f}',
             ha='center', va='top', fontsize=10)

    # Add the ΔE information
    ax2.text(0.5, 0.2, f'Color Difference: ΔE = {deltaE:.2f}',
             ha='center', va='top', fontsize=14, weight='bold',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

    # If thickness info is available, add it to the title
    title = "GST Colorimetric Analysis"
    if thicknesses is not None:
        thickness_str = ", ".join([f"{t:.1f}nm" for t in thicknesses])
        title += f"\nThicknesses: [{thickness_str}]"

    # Adjust the title position, moving it up slightly
    plt.suptitle(title, fontsize=12, weight='bold', y=0.98)  # keep y at the default 0.98
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)  # increase top margin so the title sits higher

    # Save the figure (if a path is given)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存至: {save_path}")

    plt.show()


# Usage example
if __name__ == "__main__":
    # Example Lab values (obtained from the calculation function)
    Lab_crystalline = [65.32, 12.45, -8.76]  # example values
    Lab_amorphous = [45.67, -5.43, 15.21]  # example values
    deltaE = 28.45  # example ΔE value
    thicknesses = [121, 4, 85, 39, 800]  # example thicknesses

    # Plot the comparison
    plot_gst_color_comparison(Lab_crystalline, Lab_amorphous, deltaE, thicknesses)