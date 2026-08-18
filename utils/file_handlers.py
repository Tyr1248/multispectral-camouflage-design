"""
File handling utilities
"""

import os
import json
import csv
import yaml
import numpy as np
from datetime import datetime
from PIL import Image


def _to_jsonable(obj, max_array_size=1000):
    """
    Recursively convert to JSON/YAML-serializable native Python types.

    - numpy scalars -> int/float/bool
    - small numpy arrays -> list
    - large arrays (e.g. pattern images) keep only shape info to avoid huge JSON
    """
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v, max_array_size) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v, max_array_size) for v in obj]
    if isinstance(obj, np.ndarray):
        if obj.size > max_array_size:
            return {'__ndarray__': True, 'shape': list(obj.shape), 'dtype': str(obj.dtype)}
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def save_design_results(design_params, spectrum_data, pattern_image, save_dir):
    """
    Save design results.

    Args:
        design_params: dict - design parameters
        spectrum_data: dict - spectrum data,
            structured as {'wavelengths': [...], 'colors': {color_key: {...}}}
        pattern_image: np.array / dict / None - camouflage pattern;
            pass {'amorphous': arr, 'crystalline': arr} to save both states
        save_dir: str - output directory path

    Returns:
        file_paths: dict - paths of all saved files; {} on failure
    """
    # Ensure the directory exists
    ensure_directory(save_dir)

    # Generate timestamp and unique ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    design_id = design_params.get('design_id', f'design_{timestamp}')

    file_paths = {}

    try:
        # 1. Save design parameters as JSON (handles numpy types, strips large image arrays)
        design_file = os.path.join(save_dir, f"{design_id}_params.json")
        with open(design_file, 'w', encoding='utf-8') as f:
            json.dump(_to_jsonable(design_params), f, ensure_ascii=False, indent=2)
        file_paths['design_json'] = design_file

        # 2. Save spectrum data as CSV (matches actual structure: wavelengths + colors)
        if spectrum_data and 'wavelengths' in spectrum_data and 'colors' in spectrum_data:
            wavelengths = spectrum_data['wavelengths']
            colors = spectrum_data.get('colors', {})
            color_keys = sorted(colors.keys())

            spectrum_file = os.path.join(save_dir, f"{design_id}_spectrum.csv")
            with open(spectrum_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                header = ['wavelength_nm']
                for key in color_keys:
                    header.append(f"{key}_amorphous_emissivity")
                    header.append(f"{key}_crystalline_emissivity")
                writer.writerow(header)

                for i, wl in enumerate(wavelengths):
                    row = [wl]
                    for key in color_keys:
                        color_data = colors[key]
                        for state in ('amorphous', 'crystalline'):
                            emiss = color_data.get(state, {}).get('emissivity', [])
                            # emissivity is [[per-wavelength values...]] (one row per solution); take the first
                            if emiss and len(emiss) > 0 and i < len(emiss[0]):
                                row.append(emiss[0][i])
                            else:
                                row.append('')
                    writer.writerow(row)

            file_paths['spectrum_csv'] = spectrum_file

            # 2b. Save the spectrum curve plot as PNG
            try:
                import matplotlib
                matplotlib.use('Agg')
                from matplotlib import pyplot as plt

                fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
                ax.axvspan(3000, 5000, alpha=0.15, color='lightcoral', label='3-5 μm')
                ax.axvspan(5000, 8000, alpha=0.15, color='lightgreen', label='5-8 μm')
                ax.axvspan(8000, 14000, alpha=0.15, color='lightblue', label='8-14 μm')
                for key in color_keys:
                    color_data = colors[key]
                    amorphous_emiss = color_data.get('amorphous', {}).get('emissivity', [])
                    crystalline_emiss = color_data.get('crystalline', {}).get('emissivity', [])
                    if amorphous_emiss and len(amorphous_emiss) > 0:
                        ax.plot(wavelengths, amorphous_emiss[0], '--', label=f"{key} Amorphous")
                    if crystalline_emiss and len(crystalline_emiss) > 0:
                        ax.plot(wavelengths, crystalline_emiss[0], '-', label=f"{key} Crystalline")
                ax.set_xlim(wavelengths[0], wavelengths[-1])
                ax.set_ylim(0, 1)
                ax.set_xlabel("Wavelength (nm)")
                ax.set_ylabel("Emissivity")
                ax.set_title("Infrared Emissivity Spectrum (3-14 μm)")
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
                fig.tight_layout()

                spectrum_png = os.path.join(save_dir, f"{design_id}_spectrum.png")
                fig.savefig(spectrum_png, bbox_inches="tight")
                plt.close(fig)
                file_paths['spectrum_png'] = spectrum_png
            except Exception as e:
                print(f"保存光谱图失败: {e}")

        # 3. Save pattern images (supports saving both amorphous and crystalline states)
        patterns = {}
        if isinstance(pattern_image, dict):
            patterns = {k: v for k, v in pattern_image.items() if isinstance(v, np.ndarray)}
        elif isinstance(pattern_image, np.ndarray):
            patterns = {'amorphous': pattern_image}

        for state, pattern_arr in patterns.items():
            try:
                pattern_file = os.path.join(save_dir, f"{design_id}_pattern_{state}.png")
                img = Image.fromarray(pattern_arr.astype('uint8'))
                img.save(pattern_file)
                file_paths[f'pattern_{state}'] = pattern_file
            except Exception as e:
                print(f"保存{state}图案失败: {e}")

        # 4. Generate and save preview image (amorphous/crystalline stacked, aspect ratio kept, lossless PNG)
        try:
            from core.visualization import generate_design_preview
            preview = generate_design_preview(design_params, patterns, size=(1080, 1240))
            preview_file = os.path.join(save_dir, f"{design_id}_preview.png")
            img = Image.fromarray(preview.astype('uint8'))
            img.save(preview_file)
            file_paths['preview_image'] = preview_file
        except Exception as e:
            print(f"生成预览图像失败: {e}")

        # 5. Save a summary file with all data
        summary = {
            'design_id': design_id,
            'timestamp': timestamp,
            'design_params': _to_jsonable(design_params),
            'file_paths': file_paths,
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
        import traceback
        traceback.print_exc()
        return {}


def load_design_config(config_path):
    """
    Load a design configuration.

    Args:
        config_path: str - path to the config file

    Returns:
        dict - design configuration
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
                # Try to auto-detect the format
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
    Export design data to different formats.

    Args:
        design_data: dict - design data
        export_format: str - export format

    Returns:
        bytes/str - exported data
    """
    try:
        if export_format == 'json':
            return json.dumps(design_data, ensure_ascii=False, indent=2)
        elif export_format == 'yaml' or export_format == 'yml':
            return yaml.dump(design_data, allow_unicode=True)
        elif export_format == 'csv':
            # Simplified CSV export
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
    Ensure a directory exists.

    Args:
        dir_path: str - directory path

    Returns:
        bool: whether the directory was successfully created / confirmed to exist
    """
    try:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            print(f"创建目录: {dir_path}")
        return True
    except Exception as e:
        print(f"创建目录失败: {e}")
        return False
