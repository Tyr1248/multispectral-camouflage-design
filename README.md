# 智能迷彩设计系统 (Smart Camouflage Design System)

基于 cGAN 和薄膜光学（TMM）的多层膜智能迷彩设计工具，支持多色输入、动态多解设计与静态优化设计。

## 功能概览

- **cGAN 逆向设计** — 从目标颜色反演多层薄膜结构参数（层厚、折射率）
- **TMM 光谱计算** — 矢量化传输矩阵法，支持色散多层膜与非相干叠加
- **颜色聚类与解析** — K-Harmonic Means 环境颜色提取，CIE Lab 颜色空间计算
- **迷彩图案生成** — 基于斑块数据库的数码迷彩图案渲染
- **PyQt5 图形界面** — 向导式操作流程，实时预览与结果展示

## 项目结构

```
├── main.py                  # GUI 入口
├── full_pipeline_run.py     # 命令行全流程运行
├── config.yaml              # 全局配置
├── improved__cGAN.py        # cGAN 网络定义
├── Lab_regressor.py         # Lab 颜色回归模型
├── model_utils.py           # 模型加载工具
├── core/                    # 核心引擎
│   ├── design_generation.py # 设计生成与模型推理
│   ├── color_processing.py  # 颜色提取与转换
│   ├── spectrum_calculation.py # 光谱计算
│   ├── visualization.py     # 可视化
│   └── image_analysis.py    # 环境图像分析
├── tmm_fast/                # 矢量化 TMM 物理引擎
│   ├── vectorized_tmm_dispersive_multistack.py
│   └── vectorized_incoherent_tmm.py
├── color_calculate/         # 颜色科学模块（CMF、光源、变换）
├── Cluster_extraction/      # 聚类提取（KHM、颜色特征）
├── Camo/                    # 迷彩图案生成与斑块数据库
├── ui/                      # PyQt5 界面（向导、结果窗口）
├── utils/                   # 工具函数（配置、文件处理）
├── spot_database/           # 斑块图库
├── core/materials/          # 材料折射率数据 (CSV)
├── parameters_new/          # 模型归一化参数
├── 11.14_best_model_normalized.pth   # Lab 回归器权重
└── generator_epoch100000_20251122_093538.pth  # cGAN 生成器权重
```

## 环境依赖

- Python 3.8+
- PyTorch
- PyQt5
- NumPy, SciPy, scikit-learn
- Matplotlib
- PyYAML

## 运行方式

**图形界面：**

```bash
python main.py
```

**命令行全流程：**

```bash
python full_pipeline_run.py
```

## 配置

编辑 `config.yaml` 可调整设计参数、光谱范围、迷彩类型等。

## 模型文件

两个 `.pth` 文件为训练好的网络权重，运行设计流程必需：

| 文件 | 用途 |
|---|---|
| `generator_epoch100000_20251122_093538.pth` | cGAN 生成器，输入颜色参数输出膜系结构 |
| `11.14_best_model_normalized.pth` | Lab 回归器，光谱→Lab 颜色预测 |
