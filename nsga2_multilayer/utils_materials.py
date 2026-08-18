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
    Interpolate values as a function of wavelength (NumPy implementation).

    Args:
    wls_values: NumPy array of shape (N, 2); first column is wavelength, second column is the corresponding value.
    wavelengths: 1-D NumPy array of target wavelength points to interpolate at.

    Returns:
    values_interpolated: values interpolated at the target wavelength points.
    """
    assert isinstance(wls_values, np.ndarray)
    assert isinstance(wavelengths, np.ndarray)
    assert wls_values.ndim == 2
    assert wavelengths.ndim == 1

    # Sort the input data by wavelength
    sorted_indices = np.argsort(wls_values[:, 0])
    wls_sorted = wls_values[sorted_indices, 0]
    values_sorted = wls_values[sorted_indices, 1]

    wls, values = wls_sorted, values_sorted

    # Check whether the target wavelengths are within the data range
    min_wl = np.min(wls)
    max_wl = np.max(wls)

    if np.any(wavelengths < min_wl) or np.any(wavelengths > max_wl):
        warnings.warn(
            "Extrapolation detected: Some wavelengths are outside the given data range.",
            UserWarning,
        )

    # Interpolate using NumPy's interp function
    values_interpolated = np.interp(
        wavelengths,
        wls,
        values,
        left=values[0],  # use the first value below the minimum wavelength
        right=values[-1]  # use the last value above the maximum wavelength
    )

    return values_interpolated


def interpolate_material_n_k_by_wl_numpy(material, wavelengths):
    """
    Get the refractive index (n) and extinction coefficient (k) of a material as a function of wavelength (NumPy implementation).

    Args:
    material: string, material name.
    wavelengths: 1-D NumPy array of wavelength points at which to compute the optical constants.

    Returns:
    n_material: refractive index of the material.
    k_material: extinction coefficient of the material.
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
        # Load the wavelength data directly; no conversion needed
        data_n, data_k = load_material_wavelength(material)

        n_material = interpolate_wl_numpy(data_n, wavelengths)
        k_material = interpolate_wl_numpy(data_k, wavelengths)

    return n_material, k_material


def get_n_k(materials, wl):
    """
    Get the complex refractive index (n + ik) of each layer of a multilayer stack as a function of wavelength (NumPy implementation).

    Args:
    materials: list of material names.
    wl: 1-D NumPy array of wavelength points.

    Returns:
    n_k: complex array of shape (num_wl, num_layers), the complex refractive index of each layer at each wavelength.
    """
    assert isinstance(materials, (list, np.ndarray))
    assert isinstance(wl, np.ndarray)
    assert wl.ndim == 1

    num_layers = len(materials)
    num_wl = wl.shape[0]

    # Store the results in a complex array
    n_k = np.zeros((num_layers, num_wl), dtype=np.complex128)

    for ind, material in enumerate(materials):
        n_material, k_material = interpolate_material_n_k_by_wl_numpy(material, wl)
        n_k[ind, :] = n_material + 1j * k_material


    return n_k