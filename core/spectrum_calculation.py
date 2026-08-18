import torch
import numpy as np
from tmm_fast import coh_tmm as tmm
from color_calculate.utils_materials import get_n_k


def calculate_all_spectra(designs_data, wavelength_range=(3000, 15000, 1), material_db="默认材料库"):
    """
    为每个设计结果计算晶态和非晶态对应的发射率光谱

    参数:
        designs_data: list[dict] - 简化后的设计数据
        wavelength_range: tuple - 波长计算范围 (start, end, step) nm，默认 (3000, 15000, 1)
        material_db: str - 材料数据库名称

    返回:
        dict - 光谱结果，包含波长和每个颜色/状态的发射率光谱
    """
    # 解析波长范围参数
    wl_start, wl_end, wl_step = wavelength_range

    # 生成波长数组（单位：纳米）
    wavelengths_nm = np.arange(wl_start, wl_end + wl_step, wl_step)

    # 固定材料列表结构（前4层为结构色层，第3层为GST）
    # 红外基底采用 D1 结构（Ge/ZnS 严格交替、Ge 打头，共14层，见补充材料 Table S2），
    # 最下方为熔石英基底（D1 本身不包含基底）
    materials_base = ["ZnS", "aSi", "", "SiO2",
                      "Ge", "ZnS", "Ge", "ZnS",
                      "Ge", "ZnS", "Ge", "ZnS",
                      "Ge", "ZnS", "Ge", "ZnS",
                      "Ge", "ZnS", "Fusedsilica"]

    # 固定厚度（从第5层开始，即索引4及以后）
    # D1 结构厚度 (nm): 578.4/973.6/734.0/489.2/836.6/469.6/851.8/482.5/809.7/1173.1/716.9/1192.5/668.9/688.5
    fixed_thicknesses = torch.tensor([
        578.4, 973.6, 734.0, 489.2, 836.6, 469.6, 851.8,
        482.5, 809.7, 1173.1, 716.9, 1192.5, 668.9, 688.5,
        2000000
    ], dtype=torch.float64)

    # 获取波长对应的材料光学常数
    def get_material_n_k(gst_material):
        """获取指定GST材料下的完整材料列表光学常数"""
        materials = materials_base.copy()
        materials[2] = gst_material  # 第三层为GST层
        materials = ["Air"] + materials + ["Air"]  # 添加空气层

        # 获取光学常数（输入波长单位为米）
        wavelengths_m = wavelengths_nm * 1e-9
        n_k = get_n_k(materials, wavelengths_m)

        if not isinstance(n_k, torch.Tensor):
            n_k = torch.tensor(n_k, dtype=torch.complex128)

        return n_k  # 形状为 [L, W]

    # 预计算晶态和非晶态的光学常数
    n_k_crystalline = get_material_n_k("GST_C")  # 晶态
    n_k_amorphous = get_material_n_k("GST_A")  # 非晶态

    # 定义批量计算光谱的函数
    def calculate_batch_spectra(front_thicknesses_list, n_k_batch):
        """
        批量计算多个结构的发射率光谱

        参数:
            front_thicknesses_list: list[list[float]] - 前四层厚度列表的列表
            n_k_batch: torch.Tensor - 光学常数张量，形状为 [S, L, W] 或 [L, W]

        返回:
            np.ndarray - 发射率光谱数组，形状为 [S, W]
        """
        if not front_thicknesses_list:
            return np.array([])

        S = len(front_thicknesses_list)  # 堆栈数量
        L = n_k_crystalline.shape[0]  # 层数
        W = len(wavelengths_nm)  # 波长点数

        # 构建厚度张量 [S x L]
        thicknesses_list = []
        for front_thicknesses in front_thicknesses_list:
            # 构建完整厚度列表
            full_thicknesses = torch.cat([
                torch.tensor([0.0], dtype=torch.float64),  # 第一层空气
                torch.tensor(front_thicknesses, dtype=torch.float64),  # 前四层
                fixed_thicknesses,  # 固定层
                torch.tensor([0.0], dtype=torch.float64)  # 最后一层空气
            ], dim=0)
            thicknesses_list.append(full_thicknesses)

        # 堆叠成 [S x L] 张量
        T = torch.stack(thicknesses_list, dim=0)  # 形状: [S, L]

        # 处理光学常数张量
        if n_k_batch.dim() == 2:  # 形状为 [L, W]
            # 扩展为 [S, L, W]
            N = n_k_batch.unsqueeze(0).repeat(S, 1, 1)  # 形状: [S, L, W]
        elif n_k_batch.dim() == 3:  # 形状为 [S, L, W]
            N = n_k_batch
        else:
            raise ValueError(f"Invalid n_k_batch dimension: {n_k_batch.dim()}")

        # 入射角（0度）
        Theta = torch.tensor([0.0])  # 形状: [1]

        # 波长张量（纳米）
        lambda_vacuum = torch.tensor(wavelengths_nm, dtype=torch.float64)

        # 计算TE和TM偏振的反射率和透射率
        result_te = tmm("s", N, T, Theta, lambda_vacuum)
        result_tm = tmm("p", N, T, Theta, lambda_vacuum)

        # 提取反射率和透射率
        R_TE = result_te["R"][:, 0, :]  # 形状: [S, W]
        R_TM = result_tm["R"][:, 0, :]  # 形状: [S, W]

        # 计算发射率: 1 - 反射率（TE/TM 偏振平均）
        emissivity = 1 - (R_TE + R_TM) / 2

        return emissivity.cpu().numpy()

    # 存储所有颜色的光谱结果
    all_spectra_results = {
        'wavelengths': wavelengths_nm.tolist(),
        'colors': {}
    }

    # 遍历每个颜色
    for color_design in designs_data:
        color_idx = color_design['color_index']
        color_key = f"color{color_idx + 1}"

        # 提取该颜色所有解决方案的前四层厚度
        solutions = color_design['solutions']
        front_thicknesses_list = [sol['thickness'] for sol in solutions]

        if not front_thicknesses_list:
            continue

        # 为非晶态构建批量光学常数
        n_k_amorphous_batch = n_k_amorphous.unsqueeze(0).repeat(len(front_thicknesses_list), 1, 1)
        amorphous_emissivity_batch = calculate_batch_spectra(
            front_thicknesses_list, n_k_amorphous_batch
        )

        # 为晶态构建批量光学常数
        n_k_crystalline_batch = n_k_crystalline.unsqueeze(0).repeat(len(front_thicknesses_list), 1, 1)
        crystalline_emissivity_batch = calculate_batch_spectra(
            front_thicknesses_list, n_k_crystalline_batch
        )

        # 存储该颜色的光谱数据
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

        # 添加解决方案信息
        for i, sol in enumerate(solutions):
            color_spectra['solutions_info'].append({
                'solution_type': sol.get('solution_type', 'unknown'),
                'deltaE': sol.get('deltaE', 0),
                'deltaED': sol.get('deltaED', 0),
                'thickness': sol['thickness']
            })

        all_spectra_results['colors'][color_key] = color_spectra

    return all_spectra_results