# nsga2+multilayer — 组合优化

## 概述

本项目整合了两种光学薄膜多目标优化方法，通过统一的配置系统切换。优化目标为三个红外波段（MWIR 3-5µm、RC2 5-8µm、LWIR 8-14µm）的发射率，采用 NSGA-II 或单目标 GA 搜索最优薄膜结构和材料组合。

## 项目结构

```
├── materials/                  # 光学材料 n/k 数据 (CSV)
├── tmm_fast/                   # 传递矩阵法引擎 (PyTorch GPU)
├── materials.json              # 材料索引
├── utils_materials.py          # 材料加载与插值
├── utils_units.py              # 单位转换
├── config.py                   # 优化参数配置 (模式切换)
├── cal_emissivity.py           # 批量发射率计算
├── optical_film_problem.py     # 问题编码、解码、变异、评估
├── nsga2.py                    # NSGA-II 多目标优化器
└── GA_Optimizer.py             # 单目标遗传算法优化器
```

## 两种优化模式

通过 `config.OptimizationConfig(mode=...)` 切换：

| 特性 | `periodic` (周期性结构) | `penalty` (层数惩罚) |
|------|------------------------|---------------------|
| 材料数 | 2 (Ge / ZnS) | 8 (SiO₂, ZnS, aSi, TiO₂, Ge, HfO₂, ZnSe, Al₂O₃) |
| 编码位数 | 1 bit | 3 bit |
| 相邻层约束 | 强制交替 | 不可相同 |
| 最小层数 | 5 | 3 |
| 层数惩罚权重 | 0.05 | 0.5 |
| GA 加权项 | 4 项 (含 10.6µm laser) | 3 项 |
| 发射率公式 | 1−R−T | 1−R−T |

## 快速开始

```bash
# NSGA-II 多目标优化 (默认 penalty 模式)
python nsga2.py

# 周期性结构模式
python nsga2.py periodic

# 单目标 GA 优化
python GA_Optimizer.py penalty
python GA_Optimizer.py periodic
```

## 代码中使用

```python
from config import get_default_config, get_periodic_config, get_penalty_config
from nsga2 import NSGA2Optimizer

# 层数惩罚模式
config = get_penalty_config()
config.pop_size = 200
optimizer = NSGA2Optimizer(config)
pop, o1, o2, o3, fronts = optimizer.optimize()
optimizer.save_solutions(pop, o1, o2, o3, fronts)

# 周期性结构模式
config = get_periodic_config()
optimizer = NSGA2Optimizer(config)
pop, o1, o2, o3, fronts = optimizer.optimize()
```

## 优化器对比

| | NSGA-II (`nsga2.py`) | GA (`GA_Optimizer.py`) |
|---|---|---|
| 目标 | 3 目标 (Pareto前沿) | 单目标 (Band1 最小化) |
| 选择 | 非支配排序 + 拥挤度 | 锦标赛 + 精英保留 |
| 输出 | 完整 Pareto 前沿解集 | 按适应度排序的 top-N 解 |

## 依赖

- PyTorch (GPU 加速 TMM)
- NumPy, Matplotlib, SciPy
- `tmm_fast` (项目内)
