import torch
import numpy as np
import matplotlib.pyplot as plt
from tmm_fast import coh_tmm as tmm
from color_calculate.utils_materials import get_n_k  # material utilities
import color_calculate.colors.composite as colors_composite
import colour


def calculate_gst_color_difference(thicknesses, str_mode="reflection"):
    # Validate the length of the input thickness array
    if len(thicknesses) != 4:
        raise ValueError("thicknesses参数应该包含4个厚度值：ZnS, GST_C, aSi, TiO2")

    # Define the material structures (new complex multilayer stack)
    materials_crystalline = ["Air", "ZnS", "aSi", "GST_C", "SiO2", "Ge", "Air"]
    materials_amorphous = ["Air", "ZnS", "aSi", "GST_A", "SiO2", "Ge", "Air"]

    # Define the wavelength range per ASTM E308-15 (5 nm steps, 380-780 nm)
    wavelengths_in_nm = np.arange(380, 781, 5)  # 5 nm steps
    wavelengths = wavelengths_in_nm * 1e-9  # convert to meters

    # Get the optical constants of both material configurations
    n_k_crystalline = get_n_k(materials_crystalline, wavelengths)
    n_k_amorphous = get_n_k(materials_amorphous, wavelengths)

    # Ensure n_k is a torch.Tensor
    if not isinstance(n_k_crystalline, torch.Tensor):
        n_k_crystalline = torch.tensor(n_k_crystalline, dtype=torch.complex128)
    if not isinstance(n_k_amorphous, torch.Tensor):
        n_k_amorphous = torch.tensor(n_k_amorphous, dtype=torch.complex128)

    # Stack the n_k arrays of both configurations [2 x L x W]
    n_k_stack = torch.stack([n_k_crystalline, n_k_amorphous], dim=0)

    # Define the function that computes the spectrum
    def get_spectrum(n_k_stack, thicknesses, str_mode):
        angles = torch.tensor([0.0])  # incidence angle (0 degrees = normal incidence)

        # Fixed thickness array (subsequent layers)
        fixed_thicknesses = [609.0]

        # Build the full thickness array: first four layers from the input,
        # followed by the fixed thicknesses
        if isinstance(thicknesses, torch.Tensor):
            # Convert the input thicknesses to a list
            variable_thicknesses = thicknesses.tolist()
        else:
            variable_thicknesses = thicknesses.tolist() if hasattr(thicknesses, 'tolist') else list(thicknesses)

        # Combine all thicknesses: Air(0) + first four layers (variable)
        # + subsequent layers (fixed) + Air(0)
        full_thicknesses = [0.0] + variable_thicknesses + fixed_thicknesses + [0.0]

        # Convert to a tensor
        full_thicknesses = torch.tensor(full_thicknesses)

        # Expand the thickness array to match the stack dimension of n_k_stack [S x L]
        full_thicknesses_expanded = full_thicknesses.unsqueeze(0).repeat(n_k_stack.shape[0], 1)

        # Compute reflectance and transmittance for TE and TM polarizations
        polarization_te = 's'  # TE polarization
        result_te = tmm(polarization_te, n_k_stack, full_thicknesses_expanded, angles,
                        torch.tensor(wavelengths_in_nm))

        polarization_tm = 'p'  # TM polarization
        result_tm = tmm(polarization_tm, n_k_stack, full_thicknesses_expanded, angles,
                        torch.tensor(wavelengths_in_nm))

        # Extract the results at 0-degree incidence
        R_TE = result_te['R'][:, 0, :]  # shape [S x W]
        T_TE = result_te['T'][:, 0, :]
        R_TM = result_tm['R'][:, 0, :]
        T_TM = result_tm['T'][:, 0, :]

        # Select reflection or transmission spectrum based on the mode
        if str_mode == "transmission":
            spectrum = (T_TE + T_TM) / 2  # average of TE and TM polarizations
        elif str_mode == "reflection":
            spectrum = (R_TE + R_TM) / 2
        elif str_mode == "emissivity":
            spectrum = 1 - (R_TE + R_TM) / 2 - (T_TE + T_TM) / 2
        else:
            raise ValueError(f"Invalid mode: {str_mode}")

        return spectrum

    # Compute the spectra of both states
    spectrum_stack = get_spectrum(n_k_stack, thicknesses, str_mode)

    # Split the results
    spectrum_crystalline = spectrum_stack[0]
    spectrum_amorphous = spectrum_stack[1]

    # Convert to the Lab color space
    XYZ_crystalline = colors_composite.spectrum_to_XYZ(wavelengths_in_nm, spectrum_crystalline.numpy())
    XYZ_amorphous = colors_composite.spectrum_to_XYZ(wavelengths_in_nm, spectrum_amorphous.numpy())

    Lab_crystalline = colour.XYZ_to_Lab(XYZ_crystalline)
    Lab_amorphous = colour.XYZ_to_Lab(XYZ_amorphous)

    # Compute ΔE (Euclidean distance)
    deltaE = np.sqrt(np.sum((Lab_crystalline - Lab_amorphous) ** 2))

    return Lab_crystalline, Lab_amorphous, deltaE


# Usage example
if __name__ == "__main__":
    # Example thickness array (first four layers only)
    thicknesses = torch.tensor([121, 4, 85, 39])  # ZnS, GST_C, aSi, TiO2

    # Compute the color difference
    Lab_crystalline, Lab_amorphous, deltaE = calculate_gst_color_difference(thicknesses)

    print("晶态GST的Lab值:", Lab_crystalline)
    print("非晶态GST的Lab值:", Lab_amorphous)
    print("颜色差异ΔE:", deltaE)