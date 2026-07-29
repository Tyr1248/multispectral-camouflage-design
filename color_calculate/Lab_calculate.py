import torch
import numpy as np
import matplotlib.pyplot as plt
from tmm_fast import coh_tmm as tmm
from color_calculate.utils_materials import get_n_k  # 材料工具
import color_calculate.colors.composite as colors_composite
import colour


def calculate_gst_color_difference(thicknesses, str_mode="reflection"):
    # 验证输入厚度数组的长度
    if len(thicknesses) != 4:
        raise ValueError("thicknesses参数应该包含4个厚度值：ZnS, GST_C, aSi, TiO2")

    # 定义材料结构（新的复杂多层结构）
    materials_crystalline = ["Air", "ZnS", "aSi", "GST_C", "SiO2", "Ge", "Air"]
    materials_amorphous = ["Air", "ZnS", "aSi", "GST_A", "SiO2", "Ge", "Air"]

    # 定义符合ASTM E308-15标准的波长范围（5nm间隔，380-780nm）
    wavelengths_in_nm = np.arange(380, 781, 5)  # 5nm间隔
    wavelengths = wavelengths_in_nm * 1e-9  # 转换为米

    # 获取两种材料配置的光学常数
    n_k_crystalline = get_n_k(materials_crystalline, wavelengths)
    n_k_amorphous = get_n_k(materials_amorphous, wavelengths)

    # 确保n_k是torch.Tensor类型
    if not isinstance(n_k_crystalline, torch.Tensor):
        n_k_crystalline = torch.tensor(n_k_crystalline, dtype=torch.complex128)
    if not isinstance(n_k_amorphous, torch.Tensor):
        n_k_amorphous = torch.tensor(n_k_amorphous, dtype=torch.complex128)

    # 将两种配置的n_k堆叠在一起 [2 x L x W]
    n_k_stack = torch.stack([n_k_crystalline, n_k_amorphous], dim=0)

    # 定义计算光谱的函数
    def get_spectrum(n_k_stack, thicknesses, str_mode):
        angles = torch.tensor([0.0])  # 入射角（0度表示正入射）

        # 固定厚度数组（后续层）
        fixed_thicknesses = [609.0]

        # 构建完整的厚度数组：前四层来自输入，后面是固定厚度
        if isinstance(thicknesses, torch.Tensor):
            # 将输入厚度转换为列表
            variable_thicknesses = thicknesses.tolist()
        else:
            variable_thicknesses = thicknesses.tolist() if hasattr(thicknesses, 'tolist') else list(thicknesses)

        # 组合所有厚度：Air(0) + 前四层(可变) + 后续层(固定) + Air(0)
        full_thicknesses = [0.0] + variable_thicknesses + fixed_thicknesses + [0.0]

        # 转换为张量
        full_thicknesses = torch.tensor(full_thicknesses)

        # 扩展厚度数组以匹配n_k_stack的堆栈维度 [S x L]
        full_thicknesses_expanded = full_thicknesses.unsqueeze(0).repeat(n_k_stack.shape[0], 1)

        # 计算TE和TM偏振的反射和透射率
        polarization_te = 's'  # TE偏振
        result_te = tmm(polarization_te, n_k_stack, full_thicknesses_expanded, angles,
                        torch.tensor(wavelengths_in_nm))

        polarization_tm = 'p'  # TM偏振
        result_tm = tmm(polarization_tm, n_k_stack, full_thicknesses_expanded, angles,
                        torch.tensor(wavelengths_in_nm))

        # 提取0度入射角的结果
        R_TE = result_te['R'][:, 0, :]  # 形状为 [S x W]
        T_TE = result_te['T'][:, 0, :]
        R_TM = result_tm['R'][:, 0, :]
        T_TM = result_tm['T'][:, 0, :]

        # 根据模式选择计算反射或透射光谱
        if str_mode == "transmission":
            spectrum = (T_TE + T_TM) / 2  # 平均TE和TM偏振
        elif str_mode == "reflection":
            spectrum = (R_TE + R_TM) / 2
        elif str_mode == "emissivity":
            spectrum = 1 - (R_TE + R_TM) / 2 - (T_TE + T_TM) / 2
        else:
            raise ValueError(f"Invalid mode: {str_mode}")

        return spectrum

    # 计算两种状态的光谱
    spectrum_stack = get_spectrum(n_k_stack, thicknesses, str_mode)

    # 分离结果
    spectrum_crystalline = spectrum_stack[0]
    spectrum_amorphous = spectrum_stack[1]

    # 转换为Lab颜色空间
    XYZ_crystalline = colors_composite.spectrum_to_XYZ(wavelengths_in_nm, spectrum_crystalline.numpy())
    XYZ_amorphous = colors_composite.spectrum_to_XYZ(wavelengths_in_nm, spectrum_amorphous.numpy())

    Lab_crystalline = colour.XYZ_to_Lab(XYZ_crystalline)
    Lab_amorphous = colour.XYZ_to_Lab(XYZ_amorphous)

    # 计算ΔE（欧几里得距离）
    deltaE = np.sqrt(np.sum((Lab_crystalline - Lab_amorphous) ** 2))

    return Lab_crystalline, Lab_amorphous, deltaE


# 使用示例
if __name__ == "__main__":
    # 示例厚度数组（只包含前四层厚度）
    thicknesses = torch.tensor([121, 4, 85, 39])  # ZnS, GST_C, aSi, TiO2

    # 计算颜色差异
    Lab_crystalline, Lab_amorphous, deltaE = calculate_gst_color_difference(thicknesses)

    print("晶态GST的Lab值:", Lab_crystalline)
    print("非晶态GST的Lab值:", Lab_amorphous)
    print("颜色差异ΔE:", deltaE)