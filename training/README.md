# cGAN Training

Training code and the **original dataset** for the cGAN-based multilayer
thin-film inverse design model.

## Contents

| File | Description |
|---|---|
| `train.py` | cGAN training loop (hinge loss + Lab-regression regularization, 100k epochs) |
| `dataset.py` | Dataset loading, train/val split, input normalization; saves `y_mean.npy` / `y_std.npy` |
| `cGAN.py` | Generator / Discriminator (DistributionEvaluator) definitions |
| `Lab_regressor.py` | Lab regressor network definition (pre-trained, frozen during cGAN training) |
| `lab_regressor_conv1d.py` | **Optional** — 1D-convolutional Lab regressor (alternative architecture) |
| `train_regressor.py` | Training script for the Conv1D Lab regressor |
| `dataset.csv` | Original training dataset (~62,500 rows; columns: `d1..d4` layer thicknesses [nm], `L,a,b` CIE Lab values) |

## Usage

```bash
cd training
python train.py
```

Outputs (all git-ignored):

- `weights_<timestamp>/` — best model (`best_generator_*.pth`) and periodic checkpoints
- `logs/` — per-epoch CSV training log and run configuration JSON
- `parameters/` — `y_mean.npy` / `y_std.npy` label standardization statistics
  (copy these into the project-root `parameters/` for inference)

## Prerequisites

- Python 3.8+, PyTorch (CUDA), NumPy, pandas, scikit-learn
- **A pre-trained Lab regressor** is required: `cGAN.py` loads
  `weight/lab_regressor.pth` at start-up. This weight file is
  **not included** — contact the authors to request it.

## Optional: Conv1D Lab regressor

Besides the default fully-connected regressor (`Lab_regressor.py`), an
alternative **1D-convolutional regressor** is provided
(`lab_regressor_conv1d.py`, `Conv1DModel`). You may choose either
architecture for the Lab regressor. To train the Conv1D version yourself:

```bash
python train_regressor.py   # saves the best model to weight/lab_regressor.pth
```

Note that the Conv1D regressor expects inputs of shape `(B, 1, 4)` instead of
`(B, 4)`; if you use it inside the cGAN, adapt the `lab_regressor` call in
`cGAN.py` accordingly (see `train_regressor.py` for the required
`unsqueeze(1)`).

## After training

To use your trained model in the main application:

1. Copy the generator checkpoint to `models/generator.pth`
   (or pass your own path to `load_design_model()` in `core/design_generation.py`).
2. Copy `parameters/y_mean.npy` and `y_std.npy` to the project-root `parameters/`.
3. Place the Lab regressor weights at `models/lab_regressor.pth`.
