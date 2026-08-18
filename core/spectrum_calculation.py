import torch
import numpy as np
from tmm_fast import coh_tmm as tmm
from color_calculate.utils_materials import get_n_k


def calculate_all_spectra(designs_data, wavelength_range=(3000, 15000, 1), material_db="默认材料库"):
    """
    Compute the emissivity spectra of each design in both crystalline and amorphous states

    Args:
        designs_data: list[dict] - simplified design data
        wavelength_range: tuple - wavelength range (start, end, step) in nm, default (3000, 15000, 1)
        material_db: str - material database name

    Returns:
        dict - spectral results containing the wavelengths and the emissivity spectrum of each color/state
    """
    # Parse the wavelength range parameters
    wl_start, wl_end, wl_step = wavelength_range

    # Generate the wavelength array (unit: nm)
    wavelengths_nm = np.arange(wl_start, wl_end + wl_step, wl_step)

    # Fixed material stack (the first 4 layers are the structural-color layers, the 3rd layer is GST)
    # The infrared substrate uses the D1 structure (strictly alternating Ge/ZnS starting with Ge,
    # 14 layers in total; see Table S2 in the supplementary material),
    # with a fused silica substrate at the bottom (D1 itself does not include the substrate)
    materials_base = ["ZnS", "aSi", "", "SiO2",
                      "Ge", "ZnS", "Ge", "ZnS",
                      "Ge", "ZnS", "Ge", "ZnS",
                      "Ge", "ZnS", "Ge", "ZnS",
                      "Ge", "ZnS", "Fusedsilica"]

    # Fixed thicknesses (starting from layer 5, i.e. index 4 onward)
    # D1 structure thicknesses (nm): 578.4/973.6/734.0/489.2/836.6/469.6/851.8/482.5/809.7/1173.1/716.9/1192.5/668.9/688.5
    fixed_thicknesses = torch.tensor([
        578.4, 973.6, 734.0, 489.2, 836.6, 469.6, 851.8,
        482.5, 809.7, 1173.1, 716.9, 1192.5, 668.9, 688.5,
        2000000
    ], dtype=torch.float64)

    # Get the optical constants of the materials at the given wavelengths
    def get_material_n_k(gst_material):
        """Get the optical constants of the full material stack for the given GST material"""
        materials = materials_base.copy()
        materials[2] = gst_material  # the third layer is the GST layer
        materials = ["Air"] + materials + ["Air"]  # add air layers

        # Get the optical constants (input wavelengths in meters)
        wavelengths_m = wavelengths_nm * 1e-9
        n_k = get_n_k(materials, wavelengths_m)

        if not isinstance(n_k, torch.Tensor):
            n_k = torch.tensor(n_k, dtype=torch.complex128)

        return n_k  # shape [L, W]

    # Precompute the optical constants for the crystalline and amorphous states
    n_k_crystalline = get_material_n_k("GST_C")  # crystalline
    n_k_amorphous = get_material_n_k("GST_A")  # amorphous

    # Function for computing spectra in batches
    def calculate_batch_spectra(front_thicknesses_list, n_k_batch):
        """
        Compute the emissivity spectra of multiple structures in a batch

        Args:
            front_thicknesses_list: list[list[float]] - list of front four-layer thickness lists
            n_k_batch: torch.Tensor - optical constants tensor of shape [S, L, W] or [L, W]

        Returns:
            np.ndarray - emissivity spectra of shape [S, W]
        """
        if not front_thicknesses_list:
            return np.array([])

        S = len(front_thicknesses_list)  # number of stacks
        L = n_k_crystalline.shape[0]  # number of layers
        W = len(wavelengths_nm)  # number of wavelength points

        # Build the thickness tensor [S x L]
        thicknesses_list = []
        for front_thicknesses in front_thicknesses_list:
            # Build the full thickness list
            full_thicknesses = torch.cat([
                torch.tensor([0.0], dtype=torch.float64),  # first air layer
                torch.tensor(front_thicknesses, dtype=torch.float64),  # front four layers
                fixed_thicknesses,  # fixed layers
                torch.tensor([0.0], dtype=torch.float64)  # last air layer
            ], dim=0)
            thicknesses_list.append(full_thicknesses)

        # Stack into a [S x L] tensor
        T = torch.stack(thicknesses_list, dim=0)  # shape: [S, L]

        # Process the optical constants tensor
        if n_k_batch.dim() == 2:  # shape [L, W]
            # Expand to [S, L, W]
            N = n_k_batch.unsqueeze(0).repeat(S, 1, 1)  # shape: [S, L, W]
        elif n_k_batch.dim() == 3:  # shape [S, L, W]
            N = n_k_batch
        else:
            raise ValueError(f"Invalid n_k_batch dimension: {n_k_batch.dim()}")

        # Incidence angle (0 degrees)
        Theta = torch.tensor([0.0])  # shape: [1]

        # Wavelength tensor (nm)
        lambda_vacuum = torch.tensor(wavelengths_nm, dtype=torch.float64)

        # Compute reflectance and transmittance for TE and TM polarizations
        result_te = tmm("s", N, T, Theta, lambda_vacuum)
        result_tm = tmm("p", N, T, Theta, lambda_vacuum)

        # Extract reflectance and transmittance
        R_TE = result_te["R"][:, 0, :]  # shape: [S, W]
        R_TM = result_tm["R"][:, 0, :]  # shape: [S, W]

        # Emissivity: 1 - reflectance (averaged over TE/TM polarizations)
        emissivity = 1 - (R_TE + R_TM) / 2

        return emissivity.cpu().numpy()

    # Store the spectral results for all colors
    all_spectra_results = {
        'wavelengths': wavelengths_nm.tolist(),
        'colors': {}
    }

    # Iterate over each color
    for color_design in designs_data:
        color_idx = color_design['color_index']
        color_key = f"color{color_idx + 1}"

        # Extract the front four-layer thicknesses of all solutions for this color
        solutions = color_design['solutions']
        front_thicknesses_list = [sol['thickness'] for sol in solutions]

        if not front_thicknesses_list:
            continue

        # Build the batch optical constants for the amorphous state
        n_k_amorphous_batch = n_k_amorphous.unsqueeze(0).repeat(len(front_thicknesses_list), 1, 1)
        amorphous_emissivity_batch = calculate_batch_spectra(
            front_thicknesses_list, n_k_amorphous_batch
        )

        # Build the batch optical constants for the crystalline state
        n_k_crystalline_batch = n_k_crystalline.unsqueeze(0).repeat(len(front_thicknesses_list), 1, 1)
        crystalline_emissivity_batch = calculate_batch_spectra(
            front_thicknesses_list, n_k_crystalline_batch
        )

        # Store the spectral data for this color
        color_spectra = {
            'target_rgb': color_design['target_rgb'],
            'amorphous': {
                'rgb_values': [sol['pred_rgb_amorphous'] for sol in solutions],
                'emissivity': amorphous_emissivity_batch.tolist() if amorphous_emissivity_batch.size > 0 else []
            },
            'crystalline': {
                'rgb_values': [sol['pred_rgb_crystalline'] for sol in solutions],
                'emissivity': crystalline_emissivity_batch.tolist() if crystalline_emissivity_batch.size > 0 else []
            },
            'solutions_info': []
        }

        # Add solution info
        for i, sol in enumerate(solutions):
            color_spectra['solutions_info'].append({
                'solution_type': sol.get('solution_type', 'unknown'),
                'deltaE': sol.get('deltaE', 0),
                'deltaED': sol.get('deltaED', 0),
                'thickness': sol['thickness']
            })

        all_spectra_results['colors'][color_key] = color_spectra

    return all_spectra_results
