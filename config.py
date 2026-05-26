"""
自动训练配置文件
修改此文件中的参数来控制训练行为
"""

import os

# ============================================================
# 路径配置
# ============================================================
# 数据目录
DS_DIR = r"d:\OneDrive\UCSC\Research\MMI\8um"

# 训练数据文件（已合成好的总 HDF5 文件）
TRAIN_H5 = os.path.join(DS_DIR, "train.h5")

# 模型和检查点保存目录
SAVE_DIR = os.path.join(DS_DIR, "Resnet_training_checkpoints")
os.makedirs(SAVE_DIR, exist_ok=True)

# 最佳模型保存路径
BEST_MODEL_PATH = os.path.join(SAVE_DIR, "best_model.keras")

# ============================================================
# 数据集划分参数
# ============================================================
DIVIDE_RATIO = 0.85
RANDOM_SEED = 42

# ============================================================
# 训练超参数
# ============================================================
BATCH_SIZE = 64
EPOCHS = 1000
LEARNING_RATE = 1e-4

# 早停参数
EARLY_STOPPING_PATIENCE = 100

# ReduceLROnPlateau 参数
LR_REDUCE_FACTOR = 0.5
LR_REDUCE_PATIENCE = 25
LR_MIN = 1e-6

# 绘图频率（每 N 个 epoch 保存一次训练曲线图）
PLOT_FREQUENCY = 10

# ============================================================
# 损失函数与优化器
# ============================================================
LOSS = "mse"            # "mse" 或 "mae"
METRICS = ["mse", "mae"]
OPTIMIZER = "adam"      # "adam" 或 "sgd"
SGD_MOMENTUM = 0.01     # 仅 OPTIMIZER="sgd" 时生效

# ============================================================
# GPU 配置
# ============================================================
GPU_MEMORY_GROWTH = True

# ============================================================
# 日志配置
# ============================================================
VERBOSE = 2
SUPPRESS_TF_WARNINGS = True
