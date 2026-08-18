"""
配置管理功能
"""

import os
import json
import yaml
from PyQt5.QtCore import QSettings


def load_app_config(config_file=None):
    """
    加载应用程序配置

    输入:
        config_file: str - 配置文件路径（可选）

    输出:
        dict - 应用程序配置
    """
    # 默认配置文件路径
    if config_file is None:
        config_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')

    # 如果配置文件不存在，使用默认配置
    if not os.path.exists(config_file):
        print(f"配置文件不存在，使用默认配置: {config_file}")
        return get_default_config()

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # 确保必要的配置项存在
        default_config = get_default_config()
        for key, value in default_config.items():
            if key not in config:
                config[key] = value

        print(f"Configuration file loaded: {config_file}")
        return config

    except Exception as e:
        print(f"加载配置文件时出错: {e}")
        return get_default_config()


def get_default_config():
    """
    获取默认配置

    输出:
        dict - 默认配置
    """
    return {
        'app': {
            'name': '智能迷彩设计系统',
        },
        'design': {
            'default_z_value': 20,
            'min_z_value': 1,
            'max_z_value': 100,
            'max_colors': 10,
            'default_thickness_range': [100, 500],
            'default_refractive_index_range': [1.3, 2.5],
            'dynamic_solutions_per_group': 3,
            'crystal_variation_range': 30,
            'static_crystal_variation_range': 15,
        },
        'paths': {
            'data_dir': './data',
            'results_dir': './results',
            'models_dir': './models',
        },
        'spectrum': {
            'wavelength_range': [300, 2500],
            'resolution': 1.0,
        },
        'camouflage': {
            'pattern_types': ['digital', 'woodland', 'urban', 'desert'],
            'default_pattern_type': 'digital',
            'texture_density_range': [0.1, 1.0],
        },
        'color_input': {
            'min_groups': 1,
            'max_groups': 10,
            'default_groups': 3,
        },
    }


def save_app_state(app_window):
    """
    保存应用程序状态

    输入:
        app_window: QMainWindow - 应用程序窗口对象

    输出:
        bool: 是否保存成功
    """
    try:
        settings = QSettings('国防科研单位', '智能迷彩设计系统')

        # 保存窗口几何信息
        settings.setValue('geometry', app_window.saveGeometry())
        settings.setValue('windowState', app_window.saveState())

        settings.sync()
        return True
    except Exception as e:
        print(f"保存应用程序状态时出错: {e}")
        return False


def get_default_material_properties():
    """
    获取默认材料属性

    输出:
        dict - 默认材料属性
    """
    return {
        'materials': {
            'coating_1': {
                'name': '标准涂层',
                'refractive_index_range': [1.3, 2.5],
                'thickness_range': [100, 500],
                'density': 2.5,
                'conductivity': 0.5,
                'emissivity': 0.85,
            },
            'coating_2': {
                'name': '高级涂层',
                'refractive_index_range': [1.5, 3.0],
                'thickness_range': [50, 300],
                'density': 3.0,
                'conductivity': 0.3,
                'emissivity': 0.75,
            },
        },
        'default_material': 'coating_1',
    }