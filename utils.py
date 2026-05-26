"""
数据加载工具模块
从已合成好的 train.h5 加载数据并准备 tf.data.Dataset
"""

import numpy as np
import h5py
import tensorflow as tf
from sklearn.model_selection import train_test_split


def load_data_from_h5(h5_path):
    """
    从训练 HDF5 文件加载数据。

    支持两种数据结构：
      1. 直接存储在根目录: hf['x'], hf['y']
      2. 存储在 augment 组: hf['augment/x'], hf['augment/y']

    Returns:
        x_data: (N, H, W) float32  图像数据（无通道维度）
        y_data: (N, L) float32     光谱数据
    """
    with h5py.File(h5_path, "r") as hf:
        # 尝试 augment 组
        if "augment" in hf:
            x_data = hf["augment/x"][:]
            y_data = hf["augment/y"][:]
        # 尝试根目录
        elif "x" in hf and "y" in hf:
            x_data = hf["x"][:]
            y_data = hf["y"][:]
        else:
            # 列出可用的键帮助排查
            available = list(hf.keys())
            raise KeyError(
                f"在 {h5_path} 中找不到 'x'/'y' 或 'augment/x'/'augment/y'。\n"
                f"可用的键: {available}"
            )

    print(f"✅ 加载数据: x={x_data.shape}, y={y_data.shape}")
    return x_data.astype(np.float32), y_data.astype(np.float32)


def prepare_dataset(x_data, y_data, divide_ratio=0.85, batch_size=64, random_seed=42):
    """
    划分训练/验证集并创建 tf.data.Dataset。

    Args:
        x_data: (N, H, W) 或 (N, H, W, 1)
        y_data: (N, L) 或 (N, 2, L) 或 (N, L, 1)

    Returns:
        trainset, valset, train_size, val_size
    """
    # 确保 x 有通道维度 (N, H, W, 1)
    if x_data.ndim == 3:
        x_exp = np.expand_dims(x_data, axis=-1).astype(np.float32)
    else:
        x_exp = x_data.astype(np.float32)

    # 处理 y 的形状
    if y_data.ndim == 3 and y_data.shape[1] == 2:
        # (N, 2, L) → 取第二通道（光谱值）
        y_exp = y_data[:, 1, :].astype(np.float32)
    elif y_data.ndim == 3 and y_data.shape[-1] == 1:
        y_exp = y_data.squeeze(-1).astype(np.float32)
    else:
        y_exp = y_data.astype(np.float32)

    print(f"X shape: {x_exp.shape}, dtype: {x_exp.dtype}")
    print(f"Y shape: {y_exp.shape}, dtype: {y_exp.dtype}")

    # 划分训练/验证集
    train_x, val_x, train_y, val_y = train_test_split(
        x_exp, y_exp, test_size=1 - divide_ratio, random_state=random_seed
    )

    trainset = (
        tf.data.Dataset.from_tensor_slices((train_x, train_y))
        .shuffle(len(train_x), reshuffle_each_iteration=True)
        .batch(batch_size)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )
    valset = (
        tf.data.Dataset.from_tensor_slices((val_x, val_y))
        .batch(batch_size)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )

    print(f"✅ 训练: {len(train_x)} | 验证: {len(val_x)} | batch={batch_size}")
    return trainset, valset, train_x, val_x
