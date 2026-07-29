"""
文件处理功能
"""

import os
import json
import csv
import yaml
import numpy as np
from datetime import datetime
from PIL import Image


def save_design_results(design_params, spectrum_data, pattern_image, save_dir):
    """
    保存设计结果

    输入:
        design_params: dict - 设计参数字典
        spectrum_data: dict - 光谱数据字典
        pattern_image: np.array 或 None - 迷彩图案图像数组
        save_dir: str - 保存目录路径

    输出:
        file_paths: dict - 保存的所有文件路径
    """
    # 确保目录存在
    ensure_directory(save_dir)

    # 生成时间戳和唯一ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    design_id = design_params.get('design_id', f'design_{timestamp}')

    file_paths = {}

    try:
        # 1. 保存设计参数为JSON
        design_file = os.path.join(save_dir, f"{design_id}_params.json")
        with open(design_file, 'w', encoding='utf-8') as f:
            json.dump(design_params, f, ensure_ascii=False, indent=2)
        file_paths['design_json'] = design_file

        # 2. 保存光谱数据为CSV
        if spectrum_data and 'wavelengths' in spectrum_data and 'reflectance' in spectrum_data:
            spectrum_file = os.path.join(save_dir, f"{design_id}_spectrum.csv")
            with open(spectrum_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['wavelength_nm', 'reflectance', 'absorption', 'transmittance'])

                wavelengths = spectrum_data['wavelengths']
                reflectance = spectrum_data.get('reflectance', [])
                absorption = spectrum_data.get('absorption', [])
                transmittance = spectrum_data.get('transmittance', [])

                for i in range(len(wavelengths)):
                    ref = reflectance[i] if i < len(reflectance) else 0
                    abs_val = absorption[i] if i < len(absorption) else 0
                    trans = transmittance[i] if i < len(transmittance) else 0
                    writer.writerow([wavelengths[i], ref, abs_val, trans])

            file_paths['spectrum_csv'] = spectrum_file

        # 3. 保存图案图像
        if pattern_image is not None and isinstance(pattern_image, np.ndarray):
            pattern_file = os.path.join(save_dir, f"{design_id}_pattern.png")
            img = Image.fromarray(pattern_image.astype('uint8'))
            img.save(pattern_file)
            file_paths['pattern_image'] = pattern_file

        # 4. 生成并保存预览图像
        try:
            from core.visualization import generate_design_preview
            preview = generate_design_preview(design_params, pattern_image, size=(800, 600))
            preview_file = os.path.join(save_dir, f"{design_id}_preview.jpg")
            img = Image.fromarray(preview.astype('uint8'))
            img.save(preview_file, quality=90)
            file_paths['preview_image'] = preview_file
        except Exception as e:
            print(f"生成预览图像失败: {e}")

        # 5. 保存所有数据的汇总文件
        summary = {
            'design_id': design_id,
            'timestamp': timestamp,
            'design_params': design_params,
            'file_paths': file_paths,
            'spectrum_summary': {
                'avg_reflectance': spectrum_data.get('avg_reflectance', 0) if spectrum_data else 0,
                'stealth_score': spectrum_data.get('stealth_score', 0) if spectrum_data else 0,
            } if spectrum_data else {}
        }

        summary_file = os.path.join(save_dir, f"{design_id}_summary.yaml")
        with open(summary_file, 'w', encoding='utf-8') as f:
            yaml.dump(summary, f, allow_unicode=True)
        file_paths['summary_yaml'] = summary_file

        print(f"设计结果已保存到: {save_dir}")
        print(f"设计ID: {design_id}")

        return file_paths

    except Exception as e:
        print(f"保存设计结果时出错: {e}")
        return {}


def load_design_config(config_path):
    """
    加载设计配置

    输入:
        config_path: str - 配置文件路径

    输出:
        dict - 设计配置
    """
    if not os.path.exists(config_path):
        print(f"配置文件不存在: {config_path}")
        return {}

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                return yaml.safe_load(f)
            elif config_path.endswith('.json'):
                return json.load(f)
            else:
                # 尝试自动检测格式
                content = f.read()
                try:
                    return json.loads(content)
                except:
                    return yaml.safe_load(content)
    except Exception as e:
        print(f"加载配置文件时出错: {e}")
        return {}


def export_to_format(design_data, export_format='json'):
    """
    导出设计数据为不同格式

    输入:
        design_data: dict - 设计数据
        export_format: str - 导出格式

    输出:
        bytes/str - 导出数据
    """
    try:
        if export_format == 'json':
            return json.dumps(design_data, ensure_ascii=False, indent=2)
        elif export_format == 'yaml' or export_format == 'yml':
            return yaml.dump(design_data, allow_unicode=True)
        elif export_format == 'csv':
            # 简化的CSV导出
            output = []
            for key, value in design_data.items():
                if isinstance(value, (str, int, float, bool)):
                    output.append(f"{key},{value}")
                elif isinstance(value, (list, tuple)):
                    output.append(f"{key},{','.join(map(str, value))}")
                elif isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, (str, int, float, bool)):
                            output.append(f"{key}.{subkey},{subvalue}")
            return '\n'.join(output)
        else:
            return str(design_data)
    except Exception as e:
        print(f"导出数据时出错: {e}")
        return ""


def ensure_directory(dir_path):
    """
    确保目录存在

    输入:
        dir_path: str - 目录路径

    输出:
        bool: 是否成功创建/确认目录存在
    """
    try:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            print(f"创建目录: {dir_path}")
        return True
    except Exception as e:
        print(f"创建目录失败: {e}")
        return False