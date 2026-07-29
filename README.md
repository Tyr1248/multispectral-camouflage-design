# Multispectral Camouflage Design to Camouflage Pattern

A comprehensive pipeline for on-demand multispectral camouflage design: from environment analysis and deep-learning-driven multilayer thin-film inverse design, through NSGA-II multi-objective optimization, to digital camouflage pattern generation and quantitative camouflage effectiveness evaluation.

---

## Project Overview

This repository integrates three major components:

| Component | Language | Role |
|---|---|---|
| `smart_camouflage/` (root) | Python (PyTorch) | cGAN inverse design + TMM spectrum + pattern generation |
| `camo_evaluation/` | MATLAB | Saliency-based and image-quality camouflage effectiveness evaluation |
| `nsga2_multilayer/` | Python (PyTorch) | NSGA-II / GA multi-objective optimization of multilayer thin films |

---

## Component 1 — Smart Camouflage Design (root directory)

cGAN-based inverse design of multilayer thin-film optical coatings, coupled with a vectorized Transfer Matrix Method (TMM) physics engine, color analysis, and digital camouflage pattern rendering.

### Features

- **cGAN inverse design** — predict multilayer film structure (layer thickness, refractive index) from target color
- **TMM spectrum calculation** — vectorized TMM supporting dispersive multilayer stacks and incoherent superposition
- **Color clustering & analysis** — K-Harmonic Means environment color extraction, CIE Lab color space computation
- **Camouflage pattern generation** — digital camouflage rendering from a spot database
- **PyQt5 GUI** — wizard-based workflow with real-time preview

### Structure

```
├── main.py                   # GUI entry point
├── full_pipeline_run.py      # CLI full pipeline
├── config.yaml               # Global configuration
├── improved__cGAN.py          # cGAN network definition
├── Lab_regressor.py           # Lab color regression model
├── model_utils.py             # Model loading utilities
├── core/                      # Core engine
├── tmm_fast/                  # Vectorized TMM physics engine
├── color_calculate/           # Color science (CMF, illuminants, transforms)
├── Cluster_extraction/        # Clustering (KHM, color features)
├── Camo/                      # Camouflage pattern generation & spot database
├── ui/                        # PyQt5 GUI (wizard, results windows)
├── utils/                     # Utilities (config, file handling)
├── spot_database/             # Spot pattern image database
├── parameters_new/            # Model normalization parameters
├── 11.14_best_model_normalized.pth    # Lab regressor weights
└── generator_epoch100000_20251122_093538.pth  # cGAN generator weights
```

### Dependencies

- Python 3.8+, PyTorch, PyQt5, NumPy, SciPy, scikit-learn, Matplotlib, PyYAML

### Usage

```bash
python main.py                 # GUI
python full_pipeline_run.py    # CLI full pipeline
```

---

## Component 2 — Camouflage Evaluation (`camo_evaluation/`)

MATLAB toolbox for quantitative evaluation of camouflage effectiveness, combining Graph-Based Visual Saliency (GBVS) with image quality metrics (SSIM, PSNR, CCE).

### Features

- **GBVS saliency computation** — detects regions of visual conspicuity that could compromise camouflage
- **SSIM / PSNR** — full-reference image quality assessment between camouflage and background
- **CCE (Camouflage Comprehensive Evaluation)** — feature similarity / fusion degree metric
- **Batch processing** — crop white borders, cut samples, batch SSIM/PSNR computation
- **Digital camouflage evaluation** — replace image patches with camouflage squares for A/B comparison

### Structure

```
camo_evaluation/
├── CCE/                         # Camouflage Comprehensive Evaluation (Li et al. 2025)
├── saliency_cal.m               # Region-based saliency statistics
├── compute_avg_ssim_psnr.m      # Batch SSIM/PSNR evaluation
├── replace_squares.m            # Patch replacement for camouflage testing
├── batch_crop_white_borders.m   # Batch white-border cropping
├── cut_sample.m                 # Sample cutting
├── Mark_with_redbound.m         # Mark evaluation regions
└── solid_color.m                # Solid color baseline
```

### GBVS Dependency

The saliency evaluation workflow relies on pre-computed saliency maps from the **Graph-Based Visual Saliency (GBVS)** algorithm. GBVS source code is **not included** in this repository. Download it separately from:

- [http://www.klab.caltech.edu/~harel/share/gbvs.php](http://www.klab.caltech.edu/~harel/share/gbvs.php)

See `camo_evaluation/README.md` for setup instructions.

**Reference:** J. Harel, C. Koch, and P. Perona. "Graph-Based Visual Saliency." *NIPS*, 19:545–552, 2006.

### Dependencies

- MATLAB (with Image Processing Toolbox)
- GBVS toolbox (downloaded separately)

---

## Component 3 — NSGA-II Multilayer Optimization (`nsga2_multilayer/`)

Multi-objective optimization of multilayer thin-film infrared coatings using NSGA-II and single-objective GA. Optimizes emissivity across three infrared bands (MWIR 3–5 µm, RC2 5–8 µm, LWIR 8–14 µm).

### Features

- **NSGA-II multi-objective optimization** — Pareto-front search across 3 IR bands simultaneously
- **Single-objective GA** — tournament selection with elitism for single-band minimization
- **Two operational modes** — periodic structure (Ge/ZnS alternating) and penalty-based (8 materials)
- **GPU-accelerated TMM** — PyTorch-based vectorized transfer matrix method for batch fitness evaluation

### Structure

```
nsga2_multilayer/
├── nsga2.py                    # NSGA-II multi-objective optimizer
├── GA_Optimizer.py             # Single-objective GA optimizer
├── optical_film_problem.py     # Problem encoding, decoding, mutation, evaluation
├── config.py                   # Optimization configuration (mode switching)
├── cal_emissivity.py           # Batch emissivity calculation
├── utils_materials.py          # Material loading & interpolation
├── utils_units.py              # Unit conversion
├── materials.json              # Material index
├── materials/                  # Optical material n/k data (CSV)
└── tmm_fast/                   # TMM engine (PyTorch GPU)
```

### Modes

| Feature | `periodic` | `penalty` |
|---|---|---|
| Materials | 2 (Ge / ZnS) | 8 |
| Encoding bits | 1 | 3 |
| Adjacent layer rule | Forced alternating | No consecutive same |
| Min layers | 5 | 3 |
| Layer-count penalty | 0.05 | 0.5 |

### Usage

```bash
python nsga2.py                 # NSGA-II (default: penalty mode)
python nsga2.py periodic        # NSGA-II (periodic structure mode)
python GA_Optimizer.py penalty  # Single-objective GA
python GA_Optimizer.py periodic
```

### Dependencies

- PyTorch (GPU), NumPy, Matplotlib, SciPy

---

## Third-Party Acknowledgements

This project makes use of the following open-source libraries and their associated research:

### TMM — Transfer Matrix Method

The vectorized TMM physics engine (`tmm_fast/`) is adapted from:

- [**sbyrnes321/tmm**](https://github.com/sbyrnes321/tmm) — Python TMM package by Steven Byrnes
- S. Byrnes. "Multilayer optical calculations." *arXiv:1603.02720* (2016).

### GBVS — Graph-Based Visual Saliency

The saliency evaluation workflow uses GBVS as an external dependency (not bundled):

- J. Harel, C. Koch, and P. Perona. "Graph-Based Visual Saliency." *NIPS* 2006.
- Download: [http://www.klab.caltech.edu/~harel/share/gbvs.php](http://www.klab.caltech.edu/~harel/share/gbvs.php)

### CCE — Camouflage Comprehensive Evaluation

- Y. Li, C. Jia, J. Lv, X. Qing, W. Duan, J. Zhang, and X. Weng. "Evaluation method of camouflage effect based on image feature similarity/fusion degree." *Optics & Laser Technology*, 189:113152, 2025.

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

Third-party components retain their original licenses:
- `tmm_fast/` and `gym_multilayerthinfilm/` — MIT License (adapted from [sbyrnes321/tmm](https://github.com/sbyrnes321/tmm))
- GBVS (external dependency, not bundled) — contact authors at [http://www.klab.caltech.edu/~harel/share/gbvs.php](http://www.klab.caltech.edu/~harel/share/gbvs.php)
