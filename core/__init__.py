"""
核心功能模块
"""

from .color_processing import *
from .image_analysis import *
from .design_generation import *
from .spectrum_calculation import *
from .visualization import *

__all__ = [
    # color_processing
    'extract_dominant_colors',
    'convert_color_space',
    'validate_color_values',
    'convert_multiple_colors',
    'validate_multiple_colors',
    # image_analysis
    'load_and_preprocess_image',
    'analyze_environment_texture',
    'calculate_color_statistics',
    # design_generation
    'generate_dynamic_designs',
    'generate_static_design',
    'generate_camouflage_pattern',
    # spectrum_calculation
    'calculate_infrared_spectrum',
    'calculate_visible_spectrum',
    'optimize_for_stealth',
    # visualization
    'create_color_block_image',
    'plot_spectrum_comparison',
    'generate_design_preview',
]