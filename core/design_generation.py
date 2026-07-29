import torch
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
import warnings
import random
warnings.filterwarnings('ignore')


# ================================
# 🏗️ 设计过程整合
# ================================
def load_design_model(generator_path="generator_epoch100000_20251122_093538.pth",
                      y_mean_path="parameters_new/y_mean.npy",
                      y_std_path="parameters_new/y_std.npy"):
    """
    加载生成器模型和归一化参数
    """
    # 注意：这里假设您有这些模块
    # 如果您在实际运行时遇到导入错误，请根据实际情况调整
    try:
        from improved__cGAN import Generator
        from model_utils import load_model

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载模型
        generator_model = Generator()
        generator_model = load_model(generator_model, generator_path).to(device)
        generator_model.eval()

        # 加载归一化参数
        y_mean = np.load(y_mean_path)
        y_std = np.load(y_std_path)

        return generator_model, y_mean, y_std, device
    except ImportError as e:
        print(f"警告: 无法导入设计模型模块: {e}")
        print("将使用模拟数据进行演示...")
        return None, None, None, None


def rgb_to_lab(rgb_color):
    """
    将RGB颜色转换为Lab颜色空间
    Args:
        rgb_color: RGB颜色值，范围[0, 255]
    Returns:
        lab_color: Lab颜色值
    """
    try:
        from colour import sRGB_to_XYZ, XYZ_to_Lab

        # 将RGB归一化到[0, 1]
        rgb_norm = np.array(rgb_color) / 255.0

        # 转换到XYZ
        XYZ = sRGB_to_XYZ(rgb_norm)

        # 转换到Lab
        lab = XYZ_to_Lab(XYZ)

        return lab.tolist()
    except ImportError:
        # 如果colour库不可用，使用简化转换
        print("警告: colour包不可用，使用简化的RGB到Lab转换。")
        return simplified_rgb_to_lab(rgb_color)


def simplified_rgb_to_lab(rgb_color):
    """
    简化的RGB到Lab转换（当colour库不可用时使用）
    """
    # 简化的转换方法 - 仅用于演示
    r, g, b = rgb_color

    # 归一化
    r_norm = r / 255.0
    g_norm = g / 255.0
    b_norm = b / 255.0

    # 简单的转换到Lab
    L = 116 * (0.2126 * r_norm + 0.7152 * g_norm + 0.0722 * b_norm) - 16
    a = 500 * ((0.4125 * r_norm + 0.3576 * g_norm + 0.1805 * b_norm) -
               (0.0193 * r_norm + 0.1192 * g_norm + 0.9505 * b_norm))
    b_val = 200 * ((0.0193 * r_norm + 0.1192 * g_norm + 0.9505 * b_norm) -
                   (0.0193 * r_norm + 0.1192 * g_norm + 0.9505 * b_norm))

    # 限制范围
    L = max(0, min(100, L))
    a = max(-128, min(127, a))
    b_val = max(-128, min(127, b_val))
    print("简化版rgb-lab")
    return [L, a, b_val]


def design_layers(generator_model, y_mean, y_std, device, target_lab, num_samples=1000):
    """
    设计多层结构
    Args:
        generator_model: 生成器模型
        y_mean: 归一化均值
        y_std: 归一化标准差
        device: 计算设备
        target_lab: 目标Lab颜色
        num_samples: 采样数量
    Returns:
        厚度、颜色属性等
    """
    # 如果模型不可用，生成模拟数据
    if generator_model is None:
        print("使用模拟设计数据...")
        # 生成模拟数据
        all_thicknesses = np.random.rand(num_samples, 4) * 100  # 4层结构
        all_deltaEs = np.random.rand(num_samples) * 15 + 5  # ΔE在5-20之间
        all_deltaEDs = np.random.rand(num_samples) * 20 + 10  # ΔED在10-30之间

        # 生成模拟颜色数据
        all_pred_labs_amorphous = np.random.rand(num_samples, 3) * 100
        all_pred_labs_amorphous[:, 0] = np.random.rand(num_samples) * 50 + 25  # L*在25-75之间
        all_pred_labs_amorphous[:, 1] = np.random.rand(num_samples) * 50 - 25  # a*在-25到25之间
        all_pred_labs_amorphous[:, 2] = np.random.rand(num_samples) * 50 - 25  # b*在-25到25之间

        all_pred_labs_crystalline = all_pred_labs_amorphous + np.random.randn(num_samples, 3) * 10

        return (all_thicknesses, all_deltaEs, all_deltaEDs,
                all_pred_labs_amorphous, all_pred_labs_crystalline)

    # 准备目标Lab数据
    target_lab_np = np.array(target_lab).flatten()
    labs_batch = np.tile(target_lab_np, (num_samples, 1)).astype(np.float32)

    # 生成随机z值
    z_values = torch.randn(num_samples, 2).to(device)

    # 应用归一化
    input_data = (labs_batch - y_mean) / y_std
    labs = torch.from_numpy(input_data.astype(np.float32)).to(device)

    # 生成设计
    with torch.no_grad():
        outputs = generator_model(z_values, labs)

    # 后处理
    preds = outputs.cpu().numpy()
    preds[:, 0] *= 200  # 解归一化厚度
    preds[:, 1] *= 50  # 解归一化厚度
    preds[:, 2] *= 100  # 解归一化厚度
    preds[:, 3] *= 100  # 解归一化厚度

    # 计算颜色属性
    all_thicknesses = []
    all_deltaEs = []
    all_deltaEDs = []
    all_pred_labs_amorphous = []
    all_pred_labs_crystalline = []

    for thickness in preds:
        # 计算颜色属性
        deltaE, deltaED, pred_lab_amorphous, pred_lab_crystalline = calculate_color_properties(thickness, target_lab_np)

        all_thicknesses.append(thickness)
        all_deltaEs.append(deltaE)
        all_deltaEDs.append(deltaED)
        all_pred_labs_amorphous.append(pred_lab_amorphous)
        all_pred_labs_crystalline.append(pred_lab_crystalline)

    return (np.array(all_thicknesses), np.array(all_deltaEs),
            np.array(all_deltaEDs), np.array(all_pred_labs_amorphous),
            np.array(all_pred_labs_crystalline))


def calculate_color_properties(thickness, target_lab):
    """
    计算厚度对应的颜色属性
    """
    try:
        # 这里需要导入实际的Lab计算函数
        from color_calculate import Lab_calculate

        thickness_tensor = torch.tensor(thickness, dtype=torch.float32)
        pred_lab_crystalline, pred_lab_amorphous, deltaED = Lab_calculate(thickness_tensor)

        # 计算ΔE（非晶态与目标的差异）
        deltaE = np.sqrt(np.sum((pred_lab_amorphous - target_lab) ** 2))

        return deltaE, deltaED, pred_lab_amorphous, pred_lab_crystalline
    except ImportError:
        # 模拟计算
        print("使用模拟颜色属性计算...")

        # 模拟厚度到颜色的转换
        # 这里使用简单的线性关系作为演示
        thickness_sum = np.sum(thickness)

        # 非晶态颜色：基于厚度和随机偏移
        base_color = np.array([50, 0, 0])  # 基础颜色
        thickness_factor = thickness_sum / 100  # 归一化到0-1范围
        random_offset = np.random.randn(3) * 10

        pred_lab_amorphous = base_color + thickness_factor * np.array([20, 10, 10]) + random_offset
        pred_lab_amorphous[0] = max(0, min(100, pred_lab_amorphous[0]))  # L*在0-100之间
        pred_lab_amorphous[1] = max(-50, min(50, pred_lab_amorphous[1]))  # a*在-50到50之间
        pred_lab_amorphous[2] = max(-50, min(50, pred_lab_amorphous[2]))  # b*在-50到50之间

        # 晶态颜色：基于非晶态颜色加上差异
        deltaED = np.random.rand() * 20 + 5  # ΔED在5-25之间
        color_diff = np.random.randn(3) * 5  # 随机差异
        pred_lab_crystalline = pred_lab_amorphous + color_diff
        pred_lab_crystalline[0] = max(0, min(100, pred_lab_crystalline[0]))
        pred_lab_crystalline[1] = max(-50, min(50, pred_lab_crystalline[1]))
        pred_lab_crystalline[2] = max(-50, min(50, pred_lab_crystalline[2]))

        # 计算ΔE（非晶态与目标的差异）
        deltaE = np.sqrt(np.sum((pred_lab_amorphous - target_lab) ** 2))

        return deltaE, deltaED, pred_lab_amorphous, pred_lab_crystalline


# ================================
# 🔧 聚类功能
# ================================
def filter_by_deltaE_threshold(all_thicknesses, all_deltaEs, all_deltaEDs,
                               all_pred_labs_amorphous, all_pred_labs_crystalline,
                               deltaE_threshold=5.0):
    mask = all_deltaEs <= deltaE_threshold
    return (
        all_thicknesses[mask],
        all_deltaEs[mask],
        all_deltaEDs[mask],
        all_pred_labs_amorphous[mask],
        all_pred_labs_crystalline[mask]
    )


def _cluster_kmeans_auto(thicknesses, deltaEs, deltaEDs, pred_labs_amorphous, pred_labs_crystalline,
                         min_k=2, max_k=10):
    """自动KMeans聚类"""
    colors = np.array(pred_labs_crystalline)
    if len(colors) < min_k:
        raise ValueError("Not enough samples for clustering.")

    scaler = StandardScaler()
    X = scaler.fit_transform(colors)

    # 使用轮廓系数找到最佳k值
    best_k, best_sil = min_k, -1
    k_range = range(min_k, min(max_k + 1, len(colors)))

    for k in k_range:
        try:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X)
            if len(np.unique(labels)) > 1:
                sil = silhouette_score(X, labels)
                if sil > best_sil:
                    best_sil, best_k = sil, k
        except:
            continue

    print(f"[KMeans Auto] Selected k = {best_k} (Silhouette = {best_sil:.3f})")
    final_km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = final_km.fit_predict(X)

    return _build_cluster_results(thicknesses, deltaEs, deltaEDs, pred_labs_amorphous, pred_labs_crystalline, labels, X)


def _apply_hierarchical_postprocessing(X, labels, distance_threshold=2.0, n_clusters=None):
    """
    应用层次聚类后处理，将远离簇中心的点重新分类
    Args:
        X: 标准化后的特征矩阵
        labels: 初始聚类标签
        distance_threshold: 层次聚类的距离阈值
        n_clusters: 层次聚类的簇数量（如果指定则忽略distance_threshold）
    """
    new_labels = labels.copy()
    unique_labels = np.unique(labels)

    # 处理每个非噪声簇
    for label in unique_labels:
        if label == -1:  # 跳过噪声点
            continue

        # 获取当前簇的索引
        mask = labels == label
        cluster_indices = np.where(mask)[0]

        if len(cluster_indices) < 5:  # 小簇不进行处理
            continue

        # 提取当前簇的数据
        cluster_data = X[mask]

        try:
            # 应用层次聚类
            if n_clusters is not None:
                # 基于指定簇数量进行层次聚类
                hierarchical = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    linkage='ward'
                )
            else:
                # 基于距离阈值进行层次聚类 - 关键修复：不设置n_clusters参数
                # 注意：在较新版本的scikit-learn中，设置n_clusters=None可能有问题
                # 应该完全省略这个参数，或者使用兼容的方式
                hierarchical = AgglomerativeClustering(
                    n_clusters=None,  # 显式设置为None
                    distance_threshold=distance_threshold,
                    linkage='average'
                )

            sub_labels = hierarchical.fit_predict(cluster_data)

        except ValueError as e:
            print(f"警告: 层次聚类失败 ({e})，跳过该簇的后处理")
            continue

        # 如果层次聚类产生了多个子簇，重新分配标签
        unique_sub_labels = np.unique(sub_labels)
        if len(unique_sub_labels) > 1:
            # 找到最大的子簇作为主要簇，其余作为新簇或噪声
            sub_cluster_sizes = [np.sum(sub_labels == sl) for sl in unique_sub_labels]
            main_sub_cluster = unique_sub_labels[np.argmax(sub_cluster_sizes)]

            # 重新分配标签
            for i, sub_label in enumerate(sub_labels):
                if sub_label != main_sub_cluster:
                    # 计算点到主要子簇中心的距离
                    main_cluster_mask = sub_labels == main_sub_cluster
                    main_cluster_center = np.mean(cluster_data[main_cluster_mask], axis=0)
                    point_distance = np.linalg.norm(cluster_data[i] - main_cluster_center)

                    if point_distance > distance_threshold * 2:
                        # 距离太远，标记为噪声
                        new_labels[cluster_indices[i]] = -1
                    else:
                        # 距离较近，保持为原簇（可能是边界点）
                        pass

    changed_count = np.sum(labels != new_labels)
    if changed_count > 0:
        print(f"[Hierarchical Postprocessing] Reclassified {changed_count} points")

    return new_labels


def _cluster_dbscan(thicknesses, deltaEs, deltaEDs, pred_labs_amorphous, pred_labs_crystalline,
                    eps='auto', min_samples=5, use_hierarchical_postprocessing=True):
    """DBSCAN聚类"""
    colors = np.array(pred_labs_crystalline)
    scaler = StandardScaler()
    X = scaler.fit_transform(colors)

    if eps == 'auto':
        # 自动估计eps
        nn = NearestNeighbors(n_neighbors=min_samples)
        nn.fit(X)
        distances, _ = nn.kneighbors(X)
        k_dist = np.sort(distances[:, -1])[::-1]
        eps = np.percentile(k_dist, 95)
        print(f"[DBSCAN] Estimated eps = {eps:.3f}")

    dbs = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbs.fit_predict(X)

    # 应用层次聚类后处理
    if use_hierarchical_postprocessing:
        print("Applying hierarchical clustering postprocessing...")
        try:
            labels = _apply_hierarchical_postprocessing(X, labels, distance_threshold=eps / 2)
        except Exception as e:
            print(f"层次聚类后处理失败，使用原始DBSCAN标签: {e}")

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print(f"[DBSCAN] Found {n_clusters} clusters and {n_noise} noise points.")

    return _build_cluster_results(thicknesses, deltaEs, deltaEDs, pred_labs_amorphous, pred_labs_crystalline, labels, X)


def _build_cluster_results(thicknesses, deltaEs, deltaEDs, pred_labs_amorphous, pred_labs_crystalline, labels,
                           X_scaled):
    """构建聚类结果"""
    unique_labels = [l for l in np.unique(labels) if l != -1]
    cluster_results = []

    crystalline_colors = np.array(pred_labs_crystalline)
    amorphous_colors = np.array(pred_labs_amorphous)

    for label in unique_labels:
        mask = labels == label
        best_idx_local = np.argmin(deltaEs[mask])
        global_idx = np.where(mask)[0][best_idx_local]

        # 计算非晶态颜色中心
        cluster_amorphous_colors = amorphous_colors[mask]
        amorphous_center = np.mean(cluster_amorphous_colors, axis=0).tolist()

        # 获取簇内所有厚度
        cluster_thicknesses = thicknesses[mask]
        if cluster_thicknesses.ndim == 1:
            cluster_thicknesses = cluster_thicknesses.reshape(-1, 1)

        cluster_results.append({
            'cluster_id': int(label),
            'best_thickness': thicknesses[global_idx].tolist(),
            'best_pred_lab_amorphous': pred_labs_amorphous[global_idx].tolist(),
            'best_pred_lab_crystalline': pred_labs_crystalline[global_idx].tolist(),
            'best_deltaE': float(deltaEs[global_idx]),
            'best_deltaED': float(deltaEDs[global_idx]),
            'cluster_size': int(np.sum(mask)),
            'cluster_center_color': np.mean(crystalline_colors[mask], axis=0).tolist(),
            'cluster_amorphous_center': amorphous_center,
            'cluster_deltaEs': deltaEs[mask].tolist(),
            'cluster_deltaEDs': deltaEDs[mask].tolist()
        })

    # 计算聚类指标
    valid_mask = labels != -1
    if sum(valid_mask) >= 2 and len(np.unique(labels[valid_mask])) >= 2:
        sil = silhouette_score(X_scaled[valid_mask], labels[valid_mask])
        ch = calinski_harabasz_score(X_scaled[valid_mask], labels[valid_mask])
        db = davies_bouldin_score(X_scaled[valid_mask], labels[valid_mask])
    else:
        sil, ch, db = -1, -1, -1

    metrics = {
        'silhouette_score': float(sil),
        'calinski_harabasz_score': float(ch),
        'davies_bouldin_score': float(db),
        'n_clusters': len(unique_labels),
        'n_noise': int(np.sum(labels == -1)) if -1 in labels else 0
    }

    return cluster_results, metrics


def find_max_deltaED_solution(thicknesses, deltaEs, deltaEDs, pred_labs_amorphous, pred_labs_crystalline):
    """找到ΔED最大的解"""
    max_deltaED_idx = np.argmax(deltaEDs)

    max_deltaED_solution = {
        'thickness': thicknesses[max_deltaED_idx].tolist(),
        'deltaE': float(deltaEs[max_deltaED_idx]),
        'deltaED': float(deltaEDs[max_deltaED_idx]),
        'pred_lab_amorphous': pred_labs_amorphous[max_deltaED_idx].tolist(),
        'pred_lab_crystalline': pred_labs_crystalline[max_deltaED_idx].tolist(),
        'type': 'max_deltaED'
    }

    return max_deltaED_solution


# ================================
# 🎯 精确设计函数 - 全局最优解
# ================================
def generate_accurate_design(
        target_rgb,  # 目标RGB颜色，格式如 [255, 0, 0]
        num_samples=5000,  # 采样数量（可以设置得大一些以获得更精确的结果）
        num_layers=4,  # 层数（由模型决定，这里仅供参考）
        deltaE_threshold=5.0  # ΔE阈值，用于筛选优质解
):
    """
    设计出全局最优解（ΔE最小的解），返回的数据格式和主函数相同

    Args:
        target_rgb: 目标RGB颜色 [R, G, B]，范围0-255
        num_samples: 采样数量
        num_layers: 层数
        deltaE_threshold: ΔE过滤阈值

    Returns:
        dict: 包含设计结果的信息，格式与design_and_cluster_by_color相同
    """
    print("=" * 60)
    print("🎯 精确设计 - 寻找全局最优解（ΔE最小）")
    print("=" * 60)

    # 1. RGB转Lab
    print(f"1. 目标RGB颜色: {target_rgb}")
    target_lab = rgb_to_lab(target_rgb)
    print(f"   转换到Lab颜色空间: {[round(x, 2) for x in target_lab]}")

    # 2. 加载模型
    print("\n2. 加载生成器模型...")
    generator_model, y_mean, y_std, device = load_design_model()

    if generator_model is None:
        print("   ⚠️ 使用模拟数据进行演示")
    else:
        print("   ✓ 模型加载成功")

    # 3. 生成设计
    print(f"\n3. 生成{num_samples}个设计样本...")
    all_thicknesses, all_deltaEs, all_deltaEDs, all_pred_labs_amorphous, all_pred_labs_crystalline = \
        design_layers(generator_model, y_mean, y_std, device, target_lab, num_samples)
    print(f"   ✓ 生成完成，得到{len(all_thicknesses)}个设计")

    # 4. 过滤数据（使用较严格的阈值）
    print(f"\n4. 过滤数据 (ΔE阈值 = {deltaE_threshold})...")
    filtered = filter_by_deltaE_threshold(
        all_thicknesses, all_deltaEs, all_deltaEDs,
        all_pred_labs_amorphous, all_pred_labs_crystalline,
        deltaE_threshold=deltaE_threshold
    )
    filtered_thicknesses, filtered_deltaEs, filtered_deltaEDs, \
        filtered_pred_labs_amorphous, filtered_pred_labs_crystalline = filtered

    print(f"   过滤后样本数: {len(filtered_thicknesses)} / {len(all_thicknesses)}")

    if len(filtered_thicknesses) == 0:
        print("   ⚠️ 没有找到满足ΔE阈值的解，使用所有样本")
        filtered_thicknesses = all_thicknesses
        filtered_deltaEs = all_deltaEs
        filtered_deltaEDs = all_deltaEDs
        filtered_pred_labs_amorphous = all_pred_labs_amorphous
        filtered_pred_labs_crystalline = all_pred_labs_crystalline

    # 5. 找到全局最优解（ΔE最小）
    print("\n5. 寻找全局最优解（ΔE最小）...")
    global_best_idx = np.argmin(filtered_deltaEs)
    global_best_solution = {
        'type': 'global_best',
        'thickness': filtered_thicknesses[global_best_idx].tolist(),
        'deltaE': float(filtered_deltaEs[global_best_idx]),
        'deltaED': float(filtered_deltaEDs[global_best_idx]),
        'pred_lab_amorphous': filtered_pred_labs_amorphous[global_best_idx].tolist(),
        'pred_lab_crystalline': filtered_pred_labs_crystalline[global_best_idx].tolist()
    }

    print(f"   ✓ 全局最优ΔE: {global_best_solution['deltaE']:.3f}")
    print(f"     对应ΔED: {global_best_solution['deltaED']:.3f}")

    # 6. 找到ΔED最大的解（在满足ΔE阈值的样本中）
    print("\n6. 寻找ΔED最大的解...")
    if len(filtered_deltaEDs) > 0:
        max_deltaED_idx = np.argmax(filtered_deltaEDs)
        max_deltaED_solution = {
            'type': 'max_deltaED',
            'thickness': filtered_thicknesses[max_deltaED_idx].tolist(),
            'deltaE': float(filtered_deltaEs[max_deltaED_idx]),
            'deltaED': float(filtered_deltaEDs[max_deltaED_idx]),
            'pred_lab_amorphous': filtered_pred_labs_amorphous[max_deltaED_idx].tolist(),
            'pred_lab_crystalline': filtered_pred_labs_crystalline[max_deltaED_idx].tolist()
        }
        print(f"   ✓ ΔED最大值: {max_deltaED_solution['deltaED']:.3f}")
    else:
        max_deltaED_solution = {
            'type': 'max_deltaED',
            'thickness': [],
            'deltaE': None,
            'deltaED': None,
            'pred_lab_amorphous': [],
            'pred_lab_crystalline': []
        }
        print("   ⚠️ 没有找到ΔED最大的解")

    # 7. 计算统计信息
    print("\n7. 计算统计信息...")

    # 构建最终结果
    result = {
        'target_info': {
            'target_rgb': target_rgb,
            'target_lab': target_lab,
            'num_samples': num_samples,
            'num_layers': num_layers,
            'clustering_method': 'none',  # 没有使用聚类
            'deltaE_threshold': deltaE_threshold
        },
        'clustering_metrics': None,  # 没有聚类指标
        'solutions': {
            'cluster_best': [],  # 空列表，因为没有聚类
            'global_best': global_best_solution,
            'max_deltaED': max_deltaED_solution
        },
        'statistics': {
            'total_samples': len(all_thicknesses),
            'filtered_samples': len(filtered_thicknesses),
            'num_clusters': 0,  # 没有聚类
            'avg_deltaE_filtered': float(np.mean(filtered_deltaEs)) if len(filtered_deltaEs) > 0 else None,
            'avg_deltaED_filtered': float(np.mean(filtered_deltaEDs)) if len(filtered_deltaEDs) > 0 else None,
            'min_deltaE': float(np.min(all_deltaEs)) if len(all_deltaEs) > 0 else None,
            'max_deltaED': float(np.max(all_deltaEDs)) if len(all_deltaEDs) > 0 else None,
            'min_deltaE_filtered': float(np.min(filtered_deltaEs)) if len(filtered_deltaEs) > 0 else None,
            'max_deltaED_filtered': float(np.max(filtered_deltaEDs)) if len(filtered_deltaEDs) > 0 else None
        }
    }

    print(f"\n✅ 精确设计完成!")
    print(f"   全局最优ΔE: {global_best_solution['deltaE']:.3f}")
    if max_deltaED_solution['deltaED'] is not None:
        print(f"   最大ΔED: {max_deltaED_solution['deltaED']:.3f}")
    print("=" * 60)

    return result


# ================================
# 🚀 主函数 - 设计并聚类
# ================================
def design_and_cluster_by_color(
        target_rgb,  # 目标RGB颜色，格式如 [255, 0, 0]
        num_samples=1000,  # 采样数量
        num_layers=4,  # 层数（由模型决定，这里仅供参考）
        clustering_method='dbscan',  # 聚类方法: 'kmeans_auto' 或 'dbscan'
        deltaE_threshold=10.0,  # ΔE过滤阈值
        **clustering_params  # 聚类参数
):
    """
    主函数：根据目标RGB颜色设计多层结构并进行聚类分析

    Args:
        target_rgb: 目标RGB颜色 [R, G, B]，范围0-255
        num_samples: 采样数量
        num_layers: 层数
        clustering_method: 聚类方法 ('kmeans_auto' 或 'dbscan')
        deltaE_threshold: ΔE过滤阈值
        **clustering_params: 聚类参数

    Returns:
        dict: 包含聚类结果和ΔED最大解的信息
    """
    print("=" * 60)
    print("🎨 GST多层结构颜色设计聚类分析")
    print("=" * 60)

    # 1. RGB转Lab
    print(f"1. 目标RGB颜色: {target_rgb}")
    target_lab = rgb_to_lab(target_rgb)
    print(f"   转换到Lab颜色空间: {[round(x, 2) for x in target_lab]}")

    # 2. 加载模型
    print("\n2. 加载生成器模型...")
    generator_model, y_mean, y_std, device = load_design_model()

    if generator_model is None:
        print("   ⚠️ 使用模拟数据进行演示")
    else:
        print("   ✓ 模型加载成功")

    # 3. 生成设计
    print(f"\n3. 生成{num_samples}个设计样本...")
    all_thicknesses, all_deltaEs, all_deltaEDs, all_pred_labs_amorphous, all_pred_labs_crystalline = \
        design_layers(generator_model, y_mean, y_std, device, target_lab, num_samples)
    print(f"   ✓ 生成完成，得到{len(all_thicknesses)}个设计")

    # 4. 过滤数据
    print(f"\n4. 过滤数据 (ΔE阈值 = {deltaE_threshold})...")
    filtered = filter_by_deltaE_threshold(
        all_thicknesses, all_deltaEs, all_deltaEDs,
        all_pred_labs_amorphous, all_pred_labs_crystalline,
        deltaE_threshold=deltaE_threshold
    )
    filtered_thicknesses, filtered_deltaEs, filtered_deltaEDs, \
        filtered_pred_labs_amorphous, filtered_pred_labs_crystalline = filtered

    print(f"   过滤后样本数: {len(filtered_thicknesses)} / {len(all_thicknesses)}")

    if len(filtered_thicknesses) < 5:
        print("   ⚠️ 过滤后样本过少，无法进行有效聚类")
        return None

    # 5. 聚类
    print(f"\n5. 使用{clustering_method}进行聚类...")
    try:
        if clustering_method == 'kmeans_auto':
            cluster_results, metrics = _cluster_kmeans_auto(
                filtered_thicknesses, filtered_deltaEs, filtered_deltaEDs,
                filtered_pred_labs_amorphous, filtered_pred_labs_crystalline,
                min_k=clustering_params.get('min_k', 2),
                max_k=clustering_params.get('max_k', 10)
            )
        elif clustering_method == 'dbscan':
            # 传递聚类参数，特别是层次聚类后处理参数
            use_hierarchical = clustering_params.get('use_hierarchical_postprocessing', True)
            eps_value = clustering_params.get('eps', 'auto')
            min_samples_value = clustering_params.get('min_samples', 5)

            print(f"   DBSCAN参数: eps={eps_value}, min_samples={min_samples_value}")
            print(f"   使用层次聚类后处理: {use_hierarchical}")

            cluster_results, metrics = _cluster_dbscan(
                filtered_thicknesses, filtered_deltaEs, filtered_deltaEDs,
                filtered_pred_labs_amorphous, filtered_pred_labs_crystalline,
                eps=eps_value,
                min_samples=min_samples_value,
                use_hierarchical_postprocessing=use_hierarchical
            )
        else:
            raise ValueError(f"不支持的聚类方法: {clustering_method}")

        print(f"   ✓ 聚类完成，得到{len(cluster_results)}个簇")
        print(f"   聚类指标:")
        print(f"     - 轮廓系数: {metrics['silhouette_score']:.3f}")
        print(f"     - Calinski-Harabasz指数: {metrics['calinski_harabasz_score']:.1f}")
        print(f"     - Davies-Bouldin指数: {metrics['davies_bouldin_score']:.3f}")

    except Exception as e:
        print(f"   ✗ 聚类失败: {e}")
        return None

    # 6. 找到ΔED最大的解
    print("\n6. 寻找ΔED最大的解...")
    max_deltaED_solution = find_max_deltaED_solution(
        filtered_thicknesses, filtered_deltaEs, filtered_deltaEDs,
        filtered_pred_labs_amorphous, filtered_pred_labs_crystalline
    )
    print(f"   ✓ ΔED最大值: {max_deltaED_solution['deltaED']:.3f}")

    # 7. 整理结果
    print("\n7. 整理结果...")

    # 提取各个簇的最优解
    cluster_best_solutions = []
    for cluster in cluster_results:
        solution = {
            'cluster_id': cluster['cluster_id'],
            'type': 'cluster_best',
            'thickness': cluster['best_thickness'],
            'deltaE': cluster['best_deltaE'],
            'deltaED': cluster['best_deltaED'],
            'pred_lab_amorphous': cluster['best_pred_lab_amorphous'],
            'pred_lab_crystalline': cluster['best_pred_lab_crystalline'],
            'cluster_size': cluster['cluster_size'],
            'cluster_center_color': cluster['cluster_center_color'],
            'cluster_amorphous_center': cluster['cluster_amorphous_center']
        }
        cluster_best_solutions.append(solution)

    # 全局最优解（ΔE最小）
    global_best_idx = np.argmin(filtered_deltaEs)
    global_best_solution = {
        'type': 'global_best',
        'thickness': filtered_thicknesses[global_best_idx].tolist(),
        'deltaE': float(filtered_deltaEs[global_best_idx]),
        'deltaED': float(filtered_deltaEDs[global_best_idx]),
        'pred_lab_amorphous': filtered_pred_labs_amorphous[global_best_idx].tolist(),
        'pred_lab_crystalline': filtered_pred_labs_crystalline[global_best_idx].tolist()
    }

    # 构建最终结果
    result = {
        'target_info': {
            'target_rgb': target_rgb,
            'target_lab': target_lab,
            'num_samples': num_samples,
            'num_layers': num_layers,
            'clustering_method': clustering_method,
            'deltaE_threshold': deltaE_threshold
        },
        'clustering_metrics': metrics,
        'solutions': {
            'cluster_best': cluster_best_solutions,
            'global_best': global_best_solution,
            'max_deltaED': max_deltaED_solution
        },
        'statistics': {
            'total_samples': len(all_thicknesses),
            'filtered_samples': len(filtered_thicknesses),
            'num_clusters': len(cluster_results),
            'avg_deltaE_filtered': float(np.mean(filtered_deltaEs)),
            'avg_deltaED_filtered': float(np.mean(filtered_deltaEDs))
        }
    }

    print(f"\n✅ 分析完成!")
    print(f"   共得到{len(cluster_best_solutions)}个簇的最优解")
    print(f"   全局最优ΔE: {global_best_solution['deltaE']:.3f}")
    print(f"   最大ΔED: {max_deltaED_solution['deltaED']:.3f}")
    print("=" * 60)

    return result


def print_detailed_results(result):
    """打印详细结果"""
    if result is None:
        print("没有可用的结果")
        return

    print("\n" + "=" * 60)
    print("📊 详细结果分析")
    print("=" * 60)

    # 目标信息
    target = result['target_info']
    print(f"\n🎯 目标信息:")
    print(f"   目标RGB: {target['target_rgb']}")
    print(f"   目标Lab: {[round(x, 2) for x in target['target_lab']]}")
    print(f"   采样数量: {target['num_samples']}")
    print(f"   层数: {target['num_layers']}")

    if 'clustering_method' in target:
        print(f"   聚类方法: {target['clustering_method']}")
        print(f"   ΔE阈值: {target['deltaE_threshold']}")

    # 统计信息
    stats = result['statistics']
    print(f"\n📈 统计信息:")
    print(f"   总样本数: {stats['total_samples']}")
    print(f"   过滤后样本: {stats['filtered_samples']}")

    if 'num_clusters' in stats:
        print(f"   簇数量: {stats['num_clusters']}")

    if 'avg_deltaE_filtered' in stats and stats['avg_deltaE_filtered'] is not None:
        print(f"   平均ΔE: {stats['avg_deltaE_filtered']:.3f}")

    if 'avg_deltaED_filtered' in stats and stats['avg_deltaED_filtered'] is not None:
        print(f"   平均ΔED: {stats['avg_deltaED_filtered']:.3f}")

    if 'min_deltaE' in stats and stats['min_deltaE'] is not None:
        print(f"   最小ΔE（全部）: {stats['min_deltaE']:.3f}")

    if 'max_deltaED' in stats and stats['max_deltaED'] is not None:
        print(f"   最大ΔED（全部）: {stats['max_deltaED']:.3f}")

    if 'min_deltaE_filtered' in stats and stats['min_deltaE_filtered'] is not None:
        print(f"   最小ΔE（过滤）: {stats['min_deltaE_filtered']:.3f}")

    if 'max_deltaED_filtered' in stats and stats['max_deltaED_filtered'] is not None:
        print(f"   最大ΔED（过滤）: {stats['max_deltaED_filtered']:.3f}")

    # 聚类指标（如果有）
    metrics = result['clustering_metrics']
    if metrics is not None:
        print(f"\n📊 聚类指标:")
        print(f"   轮廓系数: {metrics['silhouette_score']:.3f}")
        print(f"   Calinski-Harabasz指数: {metrics['calinski_harabasz_score']:.1f}")
        print(f"   Davies-Bouldin指数: {metrics['davies_bouldin_score']:.3f}")

    # 各簇最优解（如果有）
    solutions = result['solutions']

    if 'cluster_best' in solutions and len(solutions['cluster_best']) > 0:
        print(f"\n🏆 各簇最优解:")
        for i, sol in enumerate(solutions['cluster_best']):
            print(f"\n   簇 {sol['cluster_id']} (大小: {sol['cluster_size']}):")
            print(f"     ΔE: {sol['deltaE']:.3f}, ΔED: {sol['deltaED']:.3f}")
            print(f"     厚度: {[round(t, 2) for t in sol['thickness']]}")
            print(f"     非晶态颜色(Lab): {[round(x, 2) for x in sol['pred_lab_amorphous']]}")
            print(f"     晶态颜色(Lab): {[round(x, 2) for x in sol['pred_lab_crystalline']]}")

    # 全局最优解
    if 'global_best' in solutions:
        global_best = solutions['global_best']
        print(f"\n🥇 全局最优解 (最小ΔE):")
        print(f"   ΔE: {global_best['deltaE']:.3f}, ΔED: {global_best['deltaED']:.3f}")
        print(f"   厚度: {[round(t, 2) for t in global_best['thickness']]}")
        print(f"   非晶态颜色(Lab): {[round(x, 2) for x in global_best['pred_lab_amorphous']]}")
        print(f"   晶态颜色(Lab): {[round(x, 2) for x in global_best['pred_lab_crystalline']]}")

    # ΔED最大解
    if 'max_deltaED' in solutions:
        max_deltaED = solutions['max_deltaED']
        if max_deltaED['deltaED'] is not None:
            print(f"\n💎 ΔED最大解:")
            print(f"   ΔE: {max_deltaED['deltaE']:.3f}, ΔED: {max_deltaED['deltaED']:.3f}")
            print(f"   厚度: {[round(t, 2) for t in max_deltaED['thickness']]}")
            print(f"   非晶态颜色(Lab): {[round(x, 2) for x in max_deltaED['pred_lab_amorphous']]}")
            print(f"   晶态颜色(Lab): {[round(x, 2) for x in max_deltaED['pred_lab_crystalline']]}")

    print("\n" + "=" * 60)


# ================================
# 📋 使用示例
# ================================
if __name__ == "__main__":
    print("=" * 60)
    print("GST多层结构颜色设计系统")
    print("=" * 60)

    # 示例1: 使用精确设计函数
    print("\n1. 精确设计示例（全局最优解）")
    print("-" * 40)

    target_rgb = [155, 90, 150]  # 红色
    accurate_result = generate_accurate_design(
        target_rgb=target_rgb,
        num_samples=500,  # 使用更多样本以获得更精确的结果
        deltaE_threshold=5.0  # 较严格的ΔE阈值
    )

    print_detailed_results(accurate_result)

    # 示例2: 使用聚类设计函数
    print("\n\n2. 聚类设计示例")
    print("-" * 40)

    target_rgb = [60, 70, 165]  # 蓝色
    cluster_result = design_and_cluster_by_color(
        target_rgb=target_rgb,
        num_samples=100,
        clustering_method='kmeans_auto',
        deltaE_threshold=8.0,
        min_k=2,
        max_k=8
    )

    print_detailed_results(cluster_result)

    # 示例3: 使用DBSCAN聚类
    print("\n\n3. DBSCAN聚类设计示例")
    print("-" * 40)

    target_rgb = [100, 155, 150]  # 绿色
    dbscan_result = design_and_cluster_by_color(
        target_rgb=target_rgb,
        num_samples=150,
        clustering_method='dbscan',
        deltaE_threshold=10.0,
        min_samples=8,
        eps='auto',
        use_hierarchical_postprocessing=True
    )

    print_detailed_results(dbscan_result)

    print("\n" + "=" * 60)
    print("所有示例运行完成！")
    print("=" * 60)

def generate_camouflage_pattern(amorphous_colors, crystalline_colors, environment_features, pattern_type='digital', size=(400, 300)):
    """
    生成迷彩图案 - 使用两组颜色

    输入:
        amorphous_colors: list[tuple] - 非晶态颜色列表
        crystalline_colors: list[tuple] - 晶态颜色列表
        environment_features: dict - 环境纹理特征
        pattern_type: str - 图案类型
        size: tuple - 生成图案尺寸

    输出:
        np.array - 迷彩图案图像
        dict - 图案参数
    """
    height, width = size

    # 合并两种颜色
    all_colors = amorphous_colors + crystalline_colors

    # 创建基础图案
    pattern = np.zeros((height, width, 3), dtype=np.uint8)

    # 根据图案类型生成不同的图案
    if pattern_type == 'digital':
        # 数码迷彩：矩形块
        block_size = max(10, int(30 * environment_features.get('texture_density', 0.5)))
        for y in range(0, height, block_size):
            for x in range(0, width, block_size):
                # 交替使用非晶态和晶态颜色
                if (x // block_size + y // block_size) % 2 == 0 and amorphous_colors:
                    color_idx = (x // block_size) % len(amorphous_colors)
                    color = amorphous_colors[color_idx]
                elif crystalline_colors:
                    color_idx = (y // block_size) % len(crystalline_colors)
                    color = crystalline_colors[color_idx]
                else:
                    color = (128, 128, 128)

                block_h = min(block_size, height - y)
                block_w = min(block_size, width - x)
                pattern[y:y+block_h, x:x+block_w] = color

    else:
        # 其他图案类型使用所有颜色
        num_spots = int((width * height) / (100 * environment_features.get('texture_density', 0.5)))
        for _ in range(num_spots):
            color_idx = random.randint(0, len(all_colors) - 1)
            color = all_colors[color_idx]
            center_x = random.randint(0, width - 1)
            center_y = random.randint(0, height - 1)
            radius = random.randint(5, 20)

            # 绘制圆形斑点
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if dx*dx + dy*dy <= radius*radius:
                        x = center_x + dx
                        y = center_y + dy
                        if 0 <= x < width and 0 <= y < height:
                            pattern[y, x] = color

    # 图案参数
    pattern_params = {
        'pattern_type': pattern_type,
        'amorphous_colors_count': len(amorphous_colors),
        'crystalline_colors_count': len(crystalline_colors),
        'total_colors': len(all_colors),
        'density': environment_features.get('texture_density', 0.5),
        'blending_factor': random.uniform(0.6, 0.9),
    }

    return pattern, pattern_params


"""
新增的迷彩生成函数接口
"""


