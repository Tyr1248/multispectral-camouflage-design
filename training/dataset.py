import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os


class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def load_data(file_path, input_cols=["d1", "d2", "d3", "d4"], output_cols=["L", "a", "b"], has_header=True):
    """Load CSV data (input and output columns are configurable)."""
    if has_header:
        df = pd.read_csv(file_path)
    else:
        df = pd.read_csv(file_path, header=None)

    print("当前列名：", df.columns.tolist())

    if has_header:
        X = df[input_cols].values
        y = df[output_cols].values
    else:
        # If there is no header, use column indices (assumes input_cols and output_cols are index lists)
        X = df.iloc[:, input_cols].values
        y = df.iloc[:, output_cols].values

    return X.astype(np.float32), y.astype(np.float32)


def get_train_val_split(train_file, val_size=0.2, random_state=42,
                        has_header=True,
                        input_cols=["d1", "d2", "d3", "d4"],
                        output_cols=["L", "a", "b"],
                        param_save_dir="parameters",
                        y_mean_file="y_mean.npy",
                        y_std_file="y_std.npy"):
    """Load the training set, split it into train/validation sets, and save the standardization statistics.

    Args:
        train_file: path to the training data file
        val_size: validation set fraction
        random_state: random seed
        has_header: whether the data file has a header row
        input_cols: input column names/indices
        output_cols: output column names/indices
        param_save_dir: directory for saving parameters
        y_mean_file: filename for the y-mean parameter
        y_std_file: filename for the y-std parameter
    """
    X, y = load_data(train_file, input_cols=input_cols, output_cols=output_cols, has_header=has_header)

    # === Normalize input features ===
    # Normalize according to the thickness ranges: d1(0-150), d2(0-50), d3(0-100), d4(0-50)
    X[:, 0] = X[:, 0] / 200.0  # normalize d1 to 0-1
    X[:, 1] = X[:, 1] / 50.0  # normalize d2 to 0-1
    X[:, 2] = X[:, 2] / 100.0  # normalize d3 to 0-1
    X[:, 3] = X[:, 3] / 100.0  # normalize d4 to 0-1

    # === Standardize output labels ===
    y_mean = y.mean(axis=0)
    y_std = y.std(axis=0)
    y = (y - y_mean) / y_std  # standardize using the training set statistics

    # Create the parameter save directory (if it does not exist)
    os.makedirs(param_save_dir, exist_ok=True)

    # Save the standardization parameters (using configurable filenames)
    np.save(os.path.join(param_save_dir, y_mean_file), y_mean)
    np.save(os.path.join(param_save_dir, y_std_file), y_std)

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=val_size,
        random_state=random_state
    )

    return X_train, y_train, X_val, y_val

