"""
ResNet MMI 模型定义 v2
========================
改进:
  1. 第一个 conv stride=2 → stride=1（保留高度信息，40→40 而非 40→20）
  2. 输出 sigmoid → linear（避免峰顶/背景梯度消失）
  3. SE (Squeeze-and-Excitation) 通道注意力
  4. Gradient Loss + SAM Loss + MSE 组合损失

从 MMI 端面显微图像预测光谱（OSA）
"""

import tensorflow as tf
from tensorflow.keras import layers, models


# ============================================================
# SE 通道注意力模块
# ============================================================
def se_block(x, reduction=16):
    """Squeeze-and-Excitation: 让网络学习哪些通道更重要"""
    channels = x.shape[-1]
    # Squeeze: 全局平均池化 → 通道描述符
    se = layers.GlobalAveragePooling2D()(x)
    # Excitation: FC → ReLU → FC → Sigmoid
    se = layers.Dense(max(channels // reduction, 4), activation="relu")(se)
    se = layers.Dense(channels, activation="sigmoid")(se)
    # 恢复形状并加权
    se = layers.Reshape((1, 1, channels))(se)
    return layers.Multiply()([x, se])


# ============================================================
# 残差块（带 SE）
# ============================================================
def residual_block(x, filters, kernel_size=(3, 3), stride=1, use_se=True):
    """
    ResNet 残差块 + 可选 SE 注意力
    包含两个卷积层 + 跳跃连接
    """
    shortcut = x

    # 主路径：Conv → BN → ReLU → Conv → BN
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(filters, kernel_size, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)

    # 跳跃连接：当 stride≠1 或通道数不匹配时，调整 shortcut
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1, 1), strides=stride, padding="same")(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    # 相加 + ReLU + SE
    x = layers.Add()([x, shortcut])
    x = layers.Activation("relu")(x)
    if use_se:
        x = se_block(x)
    return x


# ============================================================
# ResNet_MMI 模型
# ============================================================
def build_resnet_mmi(input_shape=(40, 400, 1), output_length=300, use_se=True):
    """
    构建 ResNet_MMI 模型 v2

    空间变化:
        (40,400,1) → stride=1 Conv → (40,400,32)
        → rb 64,s1 → (40,400,64)
        → rb 64,s1 → (40,400,64)
        → rb 128,s2 → (20,200,128)
        → rb 128,s1 → (20,200,128)
        → rb 256,s2 → (10,100,256)
        → rb 256,s1 → (10,100,256)
        → rb 512,s2 → (5, 50,512)
        → rb 512,s1 → (5, 50,512)
        → 1×1 128 → (5,50,128) → Flatten 32000 → Dense 256 → Dense 300

    Args:
        input_shape: 输入图像形状 (H, W, C)，默认 (40, 400, 1)
        output_length: 输出光谱长度（波长点数），默认 300
        use_se: 是否使用 SE 注意力模块

    Returns:
        tf.keras.Model
    """
    inputs = layers.Input(shape=input_shape, name="input_image")

    # 初始卷积层 — stride=1 保留空间信息
    x = layers.Conv2D(32, (7, 7), strides=1, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # 残差块堆叠 (3 次 stride-2 下采样: 40→20→10→5)
    x = residual_block(x, 64,  stride=1, use_se=use_se)
    x = residual_block(x, 64,  stride=1, use_se=use_se)
    x = residual_block(x, 128, stride=2, use_se=use_se)
    x = residual_block(x, 128, stride=1, use_se=use_se)
    x = residual_block(x, 256, stride=2, use_se=use_se)
    x = residual_block(x, 256, stride=1, use_se=use_se)
    x = residual_block(x, 512, stride=2, use_se=use_se)
    x = residual_block(x, 512, stride=1, use_se=use_se)

    # 降维 → Flatten → Dense → 输出（linear 用于回归）
    x = layers.Conv2D(128, (1, 1), activation="relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(output_length, activation="linear", name="spectrum_out")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="ResNet_MMI_v2")
    return model


# ============================================================
# 损失函数
# ============================================================
def gradient_loss(y_true, y_pred):
    """
    一阶差分损失（Gradient Loss）
    惩罚光谱导数差异 → 直接优化光谱形状、峰位置、FWHM
    """
    dy_true = y_true[:, 1:] - y_true[:, :-1]
    dy_pred = y_pred[:, 1:] - y_pred[:, :-1]
    return tf.reduce_mean(tf.square(dy_pred - dy_true))


def sam_loss(y_true, y_pred):
    """
    Spectral Angle Mapper (SAM)
    衡量两条光谱的"角度"差异，对尺度不敏感
    值越小越相似，范围 [0, π]
    """
    eps = 1e-7
    dot = tf.reduce_sum(y_true * y_pred, axis=-1)
    norm_true = tf.sqrt(tf.reduce_sum(y_true ** 2, axis=-1)) + eps
    norm_pred = tf.sqrt(tf.reduce_sum(y_pred ** 2, axis=-1)) + eps
    cos_sim = tf.clip_by_value(dot / (norm_true * norm_pred), -1.0, 1.0)
    return tf.reduce_mean(tf.acos(cos_sim))


def combined_loss(y_true, y_pred, w_grad=0.1, w_sam=0.05):
    """
    组合损失:
        MSE + w_grad * GradientLoss + w_sam * SAM
    默认权重: grad=0.1, sam=0.05 (可在 Config 中调整)
    """
    mse = tf.reduce_mean(tf.square(y_pred - y_true))
    grad = gradient_loss(y_true, y_pred)
    sam  = sam_loss(y_true, y_pred)
    return mse + w_grad * grad + w_sam * sam


# ============================================================
# 自定义损失：MVLoss（可选）
# ============================================================
class MVLoss(tf.keras.losses.Loss):
    """
    自定义损失: sqrt(MSE) + sqrt(VSE)
    VSE = variance of squared errors
    """

    def __init__(self):
        super().__init__()

    def call(self, y_true, y_pred):
        mse = tf.math.reduce_mean(tf.math.square(y_pred - y_true))
        vse = tf.math.reduce_std(tf.math.square(y_pred - y_true))
        loss = tf.math.sqrt(mse) + tf.math.sqrt(vse)
        return loss
