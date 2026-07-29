import numpy as np
import warnings
from pathlib import Path
import json
import csv
import os


def load_json():
    current_dir = str(Path(__file__).parent)
    materials_file = os.path.join(current_dir, "materials.json")

    with open(materials_file, "r") as file_json:
        material_indices = json.load(file_json)

    return material_indices, current_dir


def load_material_wavelength_um(material):
    material_indices, str_directory = load_json()
    str_file = material_indices.get(material)

    if not str_file:
        raise ValueError(f"Material {material} not found in JaxLayerLumos.")

    str_csv = os.path.join(str_directory, str_file)
    data_n = []
    data_k = []

    with open(str_csv, "r") as csvfile:
        csvreader = csv.reader(csvfile)

        start_n = False
        start_k = False

        for row in csvreader:
            if len(row) == 2:
                if row[0] == "wl" and row[1] == "n":
                    start_n = True
                    start_k = False
                elif row[0] == "wl" and row[1] == "k":
                    start_n = False
                    start_k = True
                else:
                    wavelength_um, value = map(float, row)

                    if start_n and not start_k:
                        data_n.append([wavelength_um, value])
                    elif not start_n and start_k:
                        data_k.append([wavelength_um, value])
                    else:
                        raise ValueError
            elif len(row) == 0:
                pass
            else:
                raise ValueError

    data_n = np.array(data_n)
    data_k = np.array(data_k)
    assert data_n.shape[0] > 0 or data_k.shape[0] > 0

    if data_n.shape[0] == 0:
        data_n = np.concatenate(
            [data_k[:, 0][..., np.newaxis], np.zeros((data_k.shape[0], 1))], axis=1
        )
    if data_k.shape[0] == 0:
        data_k = np.concatenate(
            [data_n[:, 0][..., np.newaxis], np.zeros((data_n.shape[0], 1))], axis=1
        )

    return data_n, data_k


def load_material_wavelength(material):
    data_n, data_k = load_material_wavelength_um(material)

    data_n[:, 0] = data_n[:, 0] * 1e-6
    data_k[:, 0] = data_k[:, 0] * 1e-6

    return data_n, data_k


def interpolate_wl_numpy(wls_values, wavelengths):
    """
    基于波长进行插值计算（使用NumPy实现）

    参数:
    wls_values: 形状为(N,2)的NumPy数组，第一列为波长，第二列为对应值
    wavelengths: 一维NumPy数组，需要插值的目标波长点

    返回:
    values_interpolated: 在目标波长点插值得到的结果
    """
    assert isinstance(wls_values, np.ndarray)
    assert isinstance(wavelengths, np.ndarray)
    assert wls_values.ndim == 2
    assert wavelengths.ndim == 1

    # 对输入数据按波长排序
    sorted_indices = np.argsort(wls_values[:, 0])
    wls_sorted = wls_values[sorted_indices, 0]
    values_sorted = wls_values[sorted_indices, 1]

    wls, values = wls_sorted, values_sorted

    # 检查目标波长是否在数据范围内
    min_wl = np.min(wls)
    max_wl = np.max(wls)

    if np.any(wavelengths < min_wl) or np.any(wavelengths > max_wl):
        warnings.warn(
            "Extrapolation detected: Some wavelengths are outside the given data range.",
            UserWarning,
        )

    # 使用NumPy的interp函数进行插值
    values_interpolated = np.interp(
        wavelengths,
        wls,
        values,
        left=values[0],  # 小于最小波长时使用第一个值
        right=values[-1]  # 大于最大波长时使用最后一个值
    )

    return values_interpolated


def interpolate_material_n_k_by_wl_numpy(material, wavelengths):
    """
    基于波长获取材料的折射率(n)和消光系数(k)（使用NumPy实现）

    参数:
    material: 字符串，材料名称
    wavelengths: 一维NumPy数组，需要计算光学常数的波长点

    返回:
    n_material: 材料的折射率
    k_material: 材料的消光系数
    """
    assert isinstance(wavelengths, np.ndarray)
    assert wavelengths.ndim == 1

    if material == "Air":
        n_material = np.ones_like(wavelengths)
        k_material = np.zeros_like(wavelengths)
    elif material == "PEC":
        n_material = np.zeros_like(wavelengths) + np.inf
        k_material = np.zeros_like(wavelengths)
    else:
        # 直接加载波长数据，无需转换
        data_n, data_k = load_material_wavelength(material)

        n_material = interpolate_wl_numpy(data_n, wavelengths)
        k_material = interpolate_wl_numpy(data_k, wavelengths)

    return n_material, k_material


def get_n_k(materials, wl):
    """
    基于波长获取多层材料的复折射率(n + ik)（使用NumPy实现）

    参数:
    materials: 材料名称列表
    wl: 一维NumPy数组，波长点

    返回:
    n_k: 形状为(num_wl, num_layers)的复数数组，表示每个波长点每层材料的复折射率
    """
    assert isinstance(materials, (list, np.ndarray))
    assert isinstance(wl, np.ndarray)
    assert wl.ndim == 1

    num_layers = len(materials)
    num_wl = wl.shape[0]

    # 使用复数数组存储结果
    n_k = np.zeros((num_layers, num_wl), dtype=np.complex128)

    for ind, material in enumerate(materials):
        n_material, k_material = interpolate_material_n_k_by_wl_numpy(material, wl)
        n_k[ind, :] = n_material + 1j * k_material


    return n_k