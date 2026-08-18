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
    """加载CSV数据（输入列和输出列可配置）"""
    if has_header:
        df = pd.read_csv(file_path)
    else:
        df = pd.read_csv(file_path, header=None)

    print("当前列名：", df.columns.tolist())

    if has_header:
        X = df[input_cols].values
        y = df[output_cols].values
    else:
        # 如果没有标题，使用列索引（假设input_cols和output_cols是索引列表）
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
    """加载训练集并分割为训练集和验证集，保存标准化统计量

    Args:
        train_file: 训练数据文件路径
        val_size: 验证集比例
        random_state: 随机种子
        has_header: 数据文件是否有表头
        input_cols: 输入列名/索引
        output_cols: 输出列名/索引
        param_save_dir: 参数保存目录
        y_mean_file: y均值参数文件名
        y_std_file: y标准差参数文件名
    """
    X, y = load_data(train_file, input_cols=input_cols, output_cols=output_cols, has_header=has_header)

    # === 归一化输入特征 ===
    # 根据新的厚度范围进行归一化：d1(0-150), d2(0-50), d3(0-100), d4(0-50)
    X[:, 0] = X[:, 0] / 200.0  # d1归一化到0-1
    X[:, 1] = X[:, 1] / 50.0  # d2归一化到0-1
    X[:, 2] = X[:, 2] / 100.0  # d3归一化到0-1
    X[:, 3] = X[:, 3] / 100.0  # d4归一化到0-1

    # === 标准化输出标签 ===
    y_mean = y.mean(axis=0)
    y_std = y.std(axis=0)
    y = (y - y_mean) / y_std  # 使用训练集的统计量进行标准化

    # 创建参数保存目录（如果不存在）
    os.makedirs(param_save_dir, exist_ok=True)

    # 保存标准化参数（使用可配置的文件名）
    np.save(os.path.join(param_save_dir, y_mean_file), y_mean)
    np.save(os.path.join(param_save_dir, y_std_file), y_std)

    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=val_size,
        random_state=random_state
    )

    return X_train, y_train, X_val, y_val

