import torch
import numpy as np
from tmm_fast import coh_tmm as tmm
from utils_materials import get_n_k
import utils_units

# Set up thin-film structure and parameters
str_mode = "emissivity"

# Define wavelength range (unit: meters)
wavelengths = np.linspace(3000e-9, 14000e-9, 1100)
wavelengths_in_nm = utils_units.convert_m_to_nm(torch.tensor(wavelengths))
wavelengths_in_um = utils_units.convert_m_to_um(torch.tensor(wavelengths))


def pad_structure(materials, thicknesses, target_length):
    """Pad the materials and thicknesses lists to the target length"""
    current_length = len(materials)
    if current_length >= target_length:
        return materials[:target_length], thicknesses[:target_length]
    pad_length = target_length - current_length
    padded_materials = materials + [materials[-1]] * pad_length
    padded_thicknesses = thicknesses + [0] * pad_length
    return padded_materials, padded_thicknesses


def get_spectrum_batch(n_k_batch, thicknesses_batch, str_mode):
    """Compute spectra in batch
    n_k_batch: [B, L, W], thicknesses_batch: [B, L]
    """
    angles = torch.tensor([0.0])

    full_thicknesses_batch = torch.cat([
        torch.zeros(n_k_batch.shape[0], 1, device=thicknesses_batch.device),
        thicknesses_batch,
        torch.zeros(n_k_batch.shape[0], 1, device=thicknesses_batch.device)
    ], dim=1)

    result_te = tmm('s', n_k_batch, full_thicknesses_batch, angles, wavelengths_in_nm)
    result_tm = tmm('p', n_k_batch, full_thicknesses_batch, angles, wavelengths_in_nm)

    R_TE = result_te['R'][:, 0, :]
    T_TE = result_te['T'][:, 0, :]
    R_TM = result_tm['R'][:, 0, :]
    T_TM = result_tm['T'][:, 0, :]

    if str_mode == "transmission":
        spectrum = (T_TE + T_TM) / 2
    elif str_mode == "reflection":
        spectrum = (R_TE + R_TM) / 2
    elif str_mode == "emissivity":
        spectrum = 1 - (R_TE + R_TM) / 2 - (T_TE + T_TM) / 2
    else:
        raise ValueError(f"Invalid mode: {str_mode}")

    return spectrum


def get_emissivity_batch(spectrum_batch):
    """Compute average emissivity of each band in batch
    spectrum_batch: [B, W]
    Returns: [B, 4] — MWIR, RC2, LWIR, laser
    """
    laser_range = (10.59, 10.61)
    mwir_range = (3.0, 5.0)
    rc2_range = (5.0, 8.0)
    lwir_range = (8.0, 14.0)

    laser_idx = torch.where((wavelengths_in_um >= laser_range[0]) & (wavelengths_in_um <= laser_range[1]))[0]
    mwir_idx = torch.where((wavelengths_in_um >= mwir_range[0]) & (wavelengths_in_um <= mwir_range[1]))[0]
    rc2_idx = torch.where((wavelengths_in_um >= rc2_range[0]) & (wavelengths_in_um <= rc2_range[1]))[0]
    lwir_idx = torch.where((wavelengths_in_um >= lwir_range[0]) & (wavelengths_in_um <= lwir_range[1]))[0]

    emissivity_batch = torch.stack([
        torch.mean(spectrum_batch[:, mwir_idx], dim=1),   # MWIR
        torch.mean(spectrum_batch[:, rc2_idx], dim=1),    # RC2
        torch.mean(spectrum_batch[:, lwir_idx], dim=1),   # LWIR
        torch.mean(spectrum_batch[:, laser_idx], dim=1),  # 10.6um laser
    ], dim=1)

    return emissivity_batch


def batch_calculate_weighted_score(input_list, ga_weights=None, use_laser_term=False):
    """Compute the weighted score of multiple sets of inserted materials and thicknesses in batch

    Args:
    input_list: list of (insert_materials, insert_thicknesses)
    ga_weights: list of weights
        - 3-term mode (use_laser_term=False): [w_mwir, w_lwir_inv, w_rc2_inv]
        - 4-term mode (use_laser_term=True):  [w_mwir, w_lwir, w_rc2_inv, w_laser_inv]
    use_laser_term: whether to include the laser-band term

    Returns:
    list of [score_case1, score_case2, score_case3]
    """
    if ga_weights is None:
        ga_weights = [0.33, 0.33, 0.33]

    if not input_list:
        return []

    all_materials = []
    all_thicknesses = []

    base_configs = [
        (["Air", "ZnS", "aSi", "GST_A", "SiO2"], [100, 20, 50, 50], "A"),
        (["Air", "ZnS", "aSi", "GST_C", "SiO2"], [200, 50, 100, 100], "C"),
        (["Air", "ZnS", "aSi", "GST_C", "SiO2"], [5, 5, 5, 5], "C"),
    ]

    for insert_materials, insert_thicknesses in input_list:
        for base_materials, base_thicknesses, gst_type in base_configs:
            full_materials = base_materials + insert_materials + ["Fusedsilica", "Air"]
            full_thicknesses = base_thicknesses + insert_thicknesses + [2000000]
            all_materials.append(full_materials)
            all_thicknesses.append(full_thicknesses)

    max_layers = max(len(materials) for materials in all_materials)
    print(f"最大层数: {max_layers}")

    padded_materials = []
    padded_thicknesses = []
    for materials, thicknesses in zip(all_materials, all_thicknesses):
        padded_mat, padded_thick = pad_structure(materials, thicknesses, max_layers)
        padded_materials.append(padded_mat)
        padded_thicknesses.append(padded_thick)

    print(f"开始计算 {len(padded_materials)} 种结构的光学常数...")
    n_k_list = []
    for materials in padded_materials:
        n_k = get_n_k(materials, wavelengths)
        if not isinstance(n_k, torch.Tensor):
            n_k = torch.tensor(n_k, dtype=torch.complex128)
        n_k_list.append(n_k)

    n_k_batch = torch.stack(n_k_list, dim=0)
    thicknesses_batch = torch.tensor(padded_thicknesses, dtype=torch.float64)

    print("计算光谱...")
    spectrum_batch = get_spectrum_batch(n_k_batch, thicknesses_batch, str_mode)

    print("计算波段发射率...")
    emissivity_batch = get_emissivity_batch(spectrum_batch)  # [B, 4]: MWIR, RC2, LWIR, laser

    results = []
    num_inputs = len(input_list)

    for i in range(num_inputs):
        case1_idx = i * 3 + 0
        case2_idx = i * 3 + 1
        case3_idx = i * 3 + 2

        e1 = emissivity_batch[case1_idx]  # [MWIR, RC2, LWIR, laser]
        e2 = emissivity_batch[case2_idx]
        e3 = emissivity_batch[case3_idx]

        if use_laser_term:
            w_mwir, w_lwir, w_rc2_inv, w_laser_inv = ga_weights
            s1 = (w_mwir * e1[0] + w_lwir * e1[2] +
                  w_rc2_inv * (1 - e1[1]) + w_laser_inv * (1 - e1[3])).item()
            s2 = (w_mwir * e2[0] + w_lwir * e2[2] +
                  w_rc2_inv * (1 - e2[1]) + w_laser_inv * (1 - e2[3])).item()
            s3 = (w_mwir * e3[0] + w_lwir * e3[2] +
                  w_rc2_inv * (1 - e3[1]) + w_laser_inv * (1 - e3[3])).item()
        else:
            w_mwir, w_lwir_inv, w_rc2_inv = ga_weights
            s1 = (w_mwir * e1[0] + w_lwir_inv * (1 - e1[2]) +
                  w_rc2_inv * (1 - e1[1])).item()
            s2 = (w_mwir * e2[0] + w_lwir_inv * (1 - e2[2]) +
                  w_rc2_inv * (1 - e2[1])).item()
            s3 = (w_mwir * e3[0] + w_lwir_inv * (1 - e3[2]) +
                  w_rc2_inv * (1 - e3[1])).item()

        results.append([s1, s2, s3])

    return results
