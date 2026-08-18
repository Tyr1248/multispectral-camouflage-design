# Normalization Parameters

Place the output-label normalization parameters in this folder:

| File | Description |
|---|---|
| `y_mean.npy` | Per-channel mean of the training-set Lab labels |
| `y_std.npy` | Per-channel standard deviation of the training-set Lab labels |

These are used by `core/design_generation.py` to de-normalize the cGAN generator output.

## How to obtain

These statistics are **computed automatically from the training set** and saved into
`parameters/` when you run the training pipeline — see
[`training/`](../training/README.md) (`training/dataset.py`, `get_train_val_split()`).

Simply copy the generated `y_mean.npy` / `y_std.npy` from `training/parameters/`
into this folder, or **contact the authors** to request the files used in the paper.
