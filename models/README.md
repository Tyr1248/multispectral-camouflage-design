# Model Weights

Place the pre-trained model weights in this folder before running the application:

| File | Description |
|---|---|
| `lab_regressor.pth` | Pre-trained Lab regressor weights (loaded by the cGAN discriminator in `cGAN.py`) |
| `generator.pth` | Trained cGAN generator weights (loaded by `core/design_generation.py`) |

## How to obtain

The trained weights are **not open-sourced** with this repository. You can:

1. **Train your own generator** with the code and original dataset provided in [`training/`](../training/README.md). Note that cGAN training itself requires a pre-trained Lab regressor — see the training README for details.
2. **Contact the authors** to request the pre-trained weights (see `CITATION.cff` / the main README for contact information).

The code expects the exact file names listed above (see `cGAN.py` and `core/design_generation.py`).
