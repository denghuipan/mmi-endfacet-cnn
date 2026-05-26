"""
MMI End-facet CNN 自动训练脚本
===============================
功能：
  1. 从 train.h5 加载预处理好的数据
  2. 划分训练/验证集
  3. 构建 ResNet 模型并训练
  4. 自动保存最佳模型和训练历史

使用方式：
  python auto_train.py
  python auto_train.py --epochs 500 --batch-size 32
  python auto_train.py --data my_train.h5
"""

import os
import sys
import gc
import time
import logging
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt

from config import *
from model import build_resnet_mmi
from utils import load_data_from_h5, prepare_dataset


# ============================================================
# 训练曲线绘图回调
# ============================================================
class PlotCallback(tf.keras.callbacks.Callback):
    """每 N 个 epoch 保存训练曲线图"""

    def __init__(self, save_dir, plot_frequency=10):
        super().__init__()
        self.save_dir = save_dir
        self.plot_frequency = plot_frequency
        self.train_loss = []
        self.val_loss = []
        self.train_mse = []
        self.val_mse = []
        self.train_mae = []
        self.val_mae = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.train_loss.append(logs.get("loss"))
        self.val_loss.append(logs.get("val_loss"))
        self.train_mse.append(logs.get("mse"))
        self.val_mse.append(logs.get("val_mse"))
        self.train_mae.append(logs.get("mae", 0))
        self.val_mae.append(logs.get("val_mae", 0))

        if (epoch + 1) % self.plot_frequency == 0:
            ep = range(1, len(self.train_loss) + 1)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            ax1.plot(ep, self.train_loss, label="Train Loss")
            if self.val_loss and self.val_loss[0] is not None:
                ax1.plot(ep, self.val_loss, label="Val Loss")
            ax1.set_title(f"Loss (Epoch {epoch + 1})")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.legend()
            ax1.grid(True)

            ax2.plot(ep, self.train_mae, label="Train MAE")
            if self.val_mae and self.val_mae[0] is not None:
                ax2.plot(ep, self.val_mae, label="Val MAE")
            ax2.set_title("MAE")
            ax2.set_xlabel("Epoch")
            ax2.legend()
            ax2.grid(True)

            plt.tight_layout()
            save_path = os.path.join(self.save_dir, f"epoch_{epoch + 1}_performance.png")
            fig.savefig(save_path, dpi=100)
            plt.close(fig)
            print(f"\n[PlotCallback] saved: {save_path}")


# ============================================================
# GPU 配置
# ============================================================
def setup_gpu():
    """配置 GPU 显存"""
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if not gpus:
        print("⚠️ 未检测到 GPU，使用 CPU 训练")
        return

    print(f"GPU 数量: {len(gpus)}")
    for idx, gpu in enumerate(gpus):
        if GPU_MEMORY_GROWTH:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"  GPU {idx}: {gpu.name} ({gpu.device_type})")


# ============================================================
# 主训练流程
# ============================================================
def main():
    print("=" * 60)
    print("  MMI End-facet CNN 自动训练")
    print("=" * 60)

    # -- GPU --
    setup_gpu()

    # -- 抑制 TF 警告 --
    if SUPPRESS_TF_WARNINGS:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        tf.get_logger().setLevel(logging.ERROR)

    # -- 检查数据文件 --
    if not os.path.exists(TRAIN_H5):
        print(f"❌ 找不到训练数据文件: {TRAIN_H5}")
        print(f"   请在 config.py 中设置正确的 TRAIN_H5 路径")
        sys.exit(1)

    # -- 加载数据 --
    print(f"\n📦 加载数据: {TRAIN_H5}")
    x_data, y_data = load_data_from_h5(TRAIN_H5)

    # -- 划分数据集 --
    print(f"\n🔀 划分训练集 / 验证集...")
    trainset, valset, train_x, val_x = prepare_dataset(
        x_data, y_data,
        divide_ratio=DIVIDE_RATIO,
        batch_size=BATCH_SIZE,
        random_seed=RANDOM_SEED,
    )

    # -- 自动检测形状 --
    input_shape = (train_x.shape[1], train_x.shape[2], train_x.shape[3])
    output_len = y_data.shape[1] if y_data.ndim == 2 else y_data.shape[-1]
    print(f"\n📐 输入形状: {input_shape}")
    print(f"📐 输出长度: {output_len}")

    # -- 清理旧模型 --
    gc.collect()
    tf.keras.backend.clear_session()

    # -- 构建模型 --
    print(f"\n🏗️ 构建 ResNet 模型...")
    model = build_resnet_mmi(input_shape=input_shape, output_length=output_len)
    model.summary()
    print(f"\n总参数量: {model.count_params():,}")

    # -- 编译 --
    if OPTIMIZER.lower() == "adam":
        opt = optimizers.Adam(learning_rate=LEARNING_RATE)
    elif OPTIMIZER.lower() == "sgd":
        opt = optimizers.SGD(learning_rate=LEARNING_RATE, momentum=SGD_MOMENTUM)
    else:
        raise ValueError(f"不支持的优化器: {OPTIMIZER}")

    model.compile(optimizer=opt, loss=LOSS, metrics=METRICS)

    # -- 回调 --
    callbacks_list = [
        EarlyStopping(
            monitor="val_mse", mode="min",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True, verbose=1,
        ),
        ModelCheckpoint(
            filepath=BEST_MODEL_PATH,
            monitor="val_mse", save_best_only=True,
            mode="min", verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_mse", mode="min",
            factor=LR_REDUCE_FACTOR, patience=LR_REDUCE_PATIENCE,
            min_lr=LR_MIN, verbose=1,
        ),
        PlotCallback(save_dir=SAVE_DIR, plot_frequency=PLOT_FREQUENCY),
    ]

    # -- 训练 --
    print(f"\n🚀 开始训练")
    print(f"   Epochs: {EPOCHS}  |  Batch: {BATCH_SIZE}  |  LR: {LEARNING_RATE}")
    print(f"   EarlyStopping patience: {EARLY_STOPPING_PATIENCE}")
    print(f"   保存目录: {SAVE_DIR}")

    start_time = time.time()
    history = model.fit(
        trainset,
        validation_data=valset,
        epochs=EPOCHS,
        verbose=VERBOSE,
        callbacks=callbacks_list,
    )

    elapsed = time.time() - start_time
    print(f"\n🎉 训练完成! 耗时: {elapsed:.1f}s ({elapsed / 60:.1f}min)")

    # -- 保存 --
    if os.path.exists(BEST_MODEL_PATH):
        print(f"💾 最佳模型: {BEST_MODEL_PATH}")

    history_path = os.path.join(SAVE_DIR, "training_history.npz")
    np.savez(
        history_path,
        loss=history.history.get("loss", []),
        val_loss=history.history.get("val_loss", []),
        mse=history.history.get("mse", []),
        val_mse=history.history.get("val_mse", []),
        mae=history.history.get("mae", []),
        val_mae=history.history.get("val_mae", []),
    )
    print(f"📊 训练历史: {history_path}")

    return model, history


# ============================================================
# 命令行入口
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MMI End-facet CNN 自动训练")
    parser.add_argument("--data", type=str, default=None, help="训练数据 H5 文件路径")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    if args.data:
        TRAIN_H5 = args.data
    if args.epochs:
        EPOCHS = args.epochs
    if args.batch_size:
        BATCH_SIZE = args.batch_size
    if args.lr:
        LEARNING_RATE = args.lr

    main()
