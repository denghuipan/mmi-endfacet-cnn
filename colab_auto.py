"""
MMI End-facet CNN 全自动训练 + 测试脚本 (Colab)
================================================
一键运行: 挂载 Google Drive → 加载 train.h5 → 训练 → 加载 test.h5 → 智能报告

在 Colab 中运行:
    !python colab_auto.py

或在 notebook 中:
    %run colab_auto.py
"""

import os, sys, gc, time, logging, io
import numpy as np
import h5py
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.stats import pearsonr


# ============================================================
# 配置 —— 在这里修改参数
# ============================================================
class Config:
    # ---- Google Drive 数据路径 ----
    # 优先从环境变量读取（notebook 中设置），未设置则使用默认值
    DRIVE_MOUNT = os.environ.get("MMI_DRIVE_MOUNT", "/content/drive")
    DS_DIR      = os.environ.get("MMI_DS_DIR", "/content/drive/MyDrive/ucsc/MMI/8um")
    TRAIN_H5    = os.environ.get("MMI_TRAIN_H5", os.path.join(DS_DIR, "train.h5"))
    TEST_H5     = os.environ.get("MMI_TEST_H5",  os.path.join(DS_DIR, "single_test.h5"))
    SAVE_DIR    = os.environ.get("MMI_SAVE_DIR", os.path.join(DS_DIR, "Resnet_training_checkpoints"))
    BEST_MODEL  = os.path.join(SAVE_DIR, "best_model.keras")

    # ---- 训练超参数 ----
    DIVIDE_RATIO            = 0.85
    RANDOM_SEED             = 42
    BATCH_SIZE              = 64
    EPOCHS                  = 1000
    LEARNING_RATE           = 1e-4
    EARLY_STOPPING_PATIENCE = 100
    LR_REDUCE_FACTOR        = 0.5
    LR_REDUCE_PATIENCE      = 25
    LR_MIN                  = 1e-6
    PLOT_FREQUENCY          = 10

    # ---- 测试配置 ----
    PEAK_HEIGHT_REL     = 0.10   # 峰值检测的相对高度阈值 (相对于最大值)
    PEAK_DISTANCE       = 5      # 峰值最小间距 (波长点数)
    FIT_WINDOW          = 12     # 高斯拟合窗口 (峰值附近的点数)

    # ---- 损失函数权重 ----
    # combined_loss = MSE + w_grad * GradientLoss + w_sam * SAM
    # 设 w_grad=0 或 w_sam=0 可单独关闭对应损失
    LOSS_W_GRAD         = 0.1    # Gradient Loss 权重
    LOSS_W_SAM          = 0.05   # SAM 权重


cfg = Config()
os.makedirs(cfg.SAVE_DIR, exist_ok=True)


# ============================================================
# 1. 挂载 Google Drive
# ============================================================
def mount_drive():
    print("=" * 60)
    print("  MMI End-facet CNN 全自动训练 + 测试")
    print("=" * 60)
    if os.path.ismount(cfg.DRIVE_MOUNT):
        print("✅ Google Drive 已挂载，跳过")
        return
    print("\n🔗 挂载 Google Drive...")
    try:
        from google.colab import drive
        drive.mount(cfg.DRIVE_MOUNT)
    except ImportError:
        print("⚠️ 非 Colab 环境，跳过挂载")


# ============================================================
# 2. GPU 配置
# ============================================================
def setup_gpu():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if not gpus:
        print("⚠️ 未检测到 GPU，使用 CPU")
        return
    print(f"\n🖥️ GPU 数量: {len(gpus)}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        try:
            d = tf.config.experimental.get_device_details(gpu)
            print(f"   {gpu.name}: {d.get('device_name', 'unknown')}")
        except:
            print(f"   {gpu.name}")


# ============================================================
# 3. ResNet 模型 (v2)
# ============================================================
def se_block(x, reduction=16):
    """Squeeze-and-Excitation 通道注意力"""
    channels = x.shape[-1]
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Dense(max(channels // reduction, 4), activation="relu")(se)
    se = layers.Dense(channels, activation="sigmoid")(se)
    se = layers.Reshape((1, 1, channels))(se)
    return layers.Multiply()([x, se])


def residual_block(x, filters, kernel_size=(3, 3), stride=1, use_se=True):
    shortcut = x
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters, kernel_size, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1, 1), strides=stride, padding="same")(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    x = layers.Add()([x, shortcut])
    x = layers.Activation("relu")(x)
    if use_se:
        x = se_block(x)
    return x


def build_resnet_mmi(input_shape, output_length, use_se=True):
    """
    v2: stride=1 首层 → 保留高度信息
        输出 linear（回归） → 避免 sigmoid 梯度消失
        1×1 降维到 128 → Flatten 32000 → Dense 256
    """
    inputs = layers.Input(shape=input_shape, name="input_image")
    # stride=1 保留空间信息 (原为 stride=2)
    x = layers.Conv2D(32, (7, 7), strides=1, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = residual_block(x, 64,  stride=1, use_se=use_se)
    x = residual_block(x, 64,  stride=1, use_se=use_se)
    x = residual_block(x, 128, stride=2, use_se=use_se)
    x = residual_block(x, 128, stride=1, use_se=use_se)
    x = residual_block(x, 256, stride=2, use_se=use_se)
    x = residual_block(x, 256, stride=1, use_se=use_se)
    x = residual_block(x, 512, stride=2, use_se=use_se)
    x = residual_block(x, 512, stride=1, use_se=use_se)
    x = layers.Conv2D(128, (1, 1), activation="relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    # linear 激活 → 回归任务，避免饱和区梯度消失
    outputs = layers.Dense(output_length, activation="linear", name="spectrum_out")(x)
    return models.Model(inputs=inputs, outputs=outputs, name="ResNet_MMI_v2")


# ============================================================
# 3.5 损失函数
# ============================================================
def gradient_loss(y_true, y_pred):
    """一阶差分损失：直接优化光谱形状/峰位置/FWHM"""
    dy_true = y_true[:, 1:] - y_true[:, :-1]
    dy_pred = y_pred[:, 1:] - y_pred[:, :-1]
    return tf.reduce_mean(tf.square(dy_pred - dy_true))

def sam_loss(y_true, y_pred):
    """Spectral Angle Mapper：光谱角度损失，对尺度不敏感"""
    eps = 1e-7
    dot = tf.reduce_sum(y_true * y_pred, axis=-1)
    norm_true = tf.sqrt(tf.reduce_sum(y_true ** 2, axis=-1)) + eps
    norm_pred = tf.sqrt(tf.reduce_sum(y_pred ** 2, axis=-1)) + eps
    cos_sim = tf.clip_by_value(dot / (norm_true * norm_pred), -1.0, 1.0)
    return tf.reduce_mean(tf.acos(cos_sim))

def combined_loss(y_true, y_pred):
    """组合损失: MSE + w_grad * GradientLoss + w_sam * SAM"""
    mse = tf.reduce_mean(tf.square(y_pred - y_true))
    grad = gradient_loss(y_true, y_pred)
    sam  = sam_loss(y_true, y_pred)
    return mse + cfg.LOSS_W_GRAD * grad + cfg.LOSS_W_SAM * sam


# ============================================================
# 4. 训练曲线回调
# ============================================================
class PlotCallback(tf.keras.callbacks.Callback):
    def __init__(self, save_dir, plot_frequency=10):
        super().__init__()
        self.save_dir = save_dir
        self.plot_frequency = plot_frequency
        self.t_loss, self.v_loss = [], []
        self.t_mse,  self.v_mse  = [], []
        self.t_mae,  self.v_mae  = [], []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.t_loss.append(logs.get("loss"))
        self.v_loss.append(logs.get("val_loss"))
        self.t_mse.append(logs.get("mse"))
        self.v_mse.append(logs.get("val_mse"))
        self.t_mae.append(logs.get("mae", 0))
        self.v_mae.append(logs.get("val_mae", 0))
        if (epoch + 1) % self.plot_frequency == 0:
            ep = range(1, len(self.t_loss) + 1)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            ax1.plot(ep, self.t_loss, label="Train")
            if self.v_loss[0] is not None: ax1.plot(ep, self.v_loss, label="Val")
            ax1.set_title(f"Loss (Epoch {epoch+1})"); ax1.legend(); ax1.grid(True)
            ax2.plot(ep, self.t_mae, label="Train")
            if self.v_mae[0] is not None: ax2.plot(ep, self.v_mae, label="Val")
            ax2.set_title("MAE"); ax2.legend(); ax2.grid(True)
            plt.tight_layout()
            p = os.path.join(self.save_dir, f"epoch_{epoch+1}_performance.png")
            fig.savefig(p, dpi=100); plt.close(fig)
            print(f"\n[Plot] {p}")


# ============================================================
# 5. 数据加载
# ============================================================
def load_data(h5_path):
    print(f"\n📦 加载: {h5_path}")
    with h5py.File(h5_path, "r") as hf:
        if "augment" in hf:
            x, y = hf["augment/x"][:], hf["augment/y"][:]
        elif "x" in hf and "y" in hf:
            x, y = hf["x"][:], hf["y"][:]
        else:
            raise KeyError(f"找不到数据。可用键: {list(hf.keys())}")
    print(f"   x={x.shape}, y={y.shape}")
    return x.astype(np.float32), y.astype(np.float32)


def prepare(x_data, y_data):
    if x_data.ndim == 3:
        x = np.expand_dims(x_data, -1).astype(np.float32)
    else:
        x = x_data.astype(np.float32)

    if y_data.ndim == 3 and y_data.shape[1] == 2:
        y = y_data[:, 1, :].astype(np.float32)
    elif y_data.ndim == 3 and y_data.shape[-1] == 1:
        y = y_data.squeeze(-1).astype(np.float32)
    else:
        y = y_data.astype(np.float32)

    print(f"   X: {x.shape}  Y: {y.shape}")

    tx, vx, ty, vy = train_test_split(
        x, y, test_size=1 - cfg.DIVIDE_RATIO, random_state=cfg.RANDOM_SEED
    )
    ds_train = (tf.data.Dataset.from_tensor_slices((tx, ty))
                .shuffle(len(tx), reshuffle_each_iteration=True)
                .batch(cfg.BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE))
    ds_val = (tf.data.Dataset.from_tensor_slices((vx, vy))
              .batch(cfg.BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE))
    print(f"   训练: {len(tx)}  验证: {len(vx)}  batch={cfg.BATCH_SIZE}")
    return ds_train, ds_val, tx, vx


# ============================================================
# 6. 测试数据加载
# ============================================================
def load_test_data(h5_path):
    """
    加载测试 H5 文件。
    测试文件结构:
        img          (N, H, W)      图像
        osa          (N, L) 或 (N, L, 1)  光谱
        wavelength   (L,)           波长数组
        labels       (N,)           标签
    """
    print(f"\n📦 加载测试数据: {h5_path}")
    with h5py.File(h5_path, "r") as hf:
        keys = list(hf.keys())
        print(f"   可用键: {keys}")
        img = hf["img"][:].astype(np.float32)
        osa = hf["osa"][:].astype(np.float32)
        wl  = hf["wavelength"][:].astype(np.float32)
        lbl = [s.decode("utf-8") if isinstance(s, bytes) else str(s)
               for s in hf["labels"][:]]
    if img.ndim == 3:
        img = np.expand_dims(img, -1)
    if osa.ndim == 3:
        osa = osa.squeeze(-1)
    print(f"   img: {img.shape}  osa: {osa.shape}  wl: {wl.shape}  labels: {len(lbl)}")
    return img, osa, wl, lbl


# ============================================================
# 7. 峰值检测与测试类型分类
# ============================================================
def gaussian_fn(x, amp, mu, sigma, offset):
    return amp * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) + offset


def detect_peaks_per_sample(spectrum, wavelengths, height_rel=0.10, distance=5):
    """对单个光谱做峰值检测，返回峰值波长和属性"""
    norm = spectrum / (spectrum.max() + 1e-10)
    peaks_idx, props = find_peaks(norm, height=height_rel, distance=distance)
    peak_wls = wavelengths[peaks_idx]
    peak_heights = props["peak_heights"]
    return peak_wls, peak_heights, peaks_idx


def auto_detect_test_type(y_true, wavelengths, height_rel=0.10, distance=5):
    """
    根据所有样本的平均峰值数判断测试类型:
        broad  (broad spectrum): 平均 < 0.5 个峰
        single (single peak)   : 平均 0.5~1.5 个峰
        dual   (dual peak)     : 平均 1.5~2.5 个峰
        multi  (multi peak)    : 平均 ≥ 2.5 个峰
    """
    n_peaks_list = []
    for i in range(len(y_true)):
        peaks, _, _ = detect_peaks_per_sample(
            y_true[i], wavelengths, height_rel=height_rel, distance=distance
        )
        n_peaks_list.append(len(peaks))
    avg_peaks = np.mean(n_peaks_list)
    print(f"\n🔍 平均峰值数: {avg_peaks:.1f}")
    if avg_peaks < 0.5:
        return "broad"
    elif avg_peaks < 1.5:
        return "single"
    elif avg_peaks < 2.5:
        return "dual"
    else:
        return "multi"


# ============================================================
# 8. 报告生成
# ============================================================
def _fit_gaussian_to_peak(wavelengths, spectrum, peak_idx, window=12):
    """在峰值附近做高斯拟合，返回 (amp, mu, sigma, fwhm, r2)"""
    half = window // 2
    lo, hi = max(0, peak_idx - half), min(len(spectrum), peak_idx + half + 1)
    x_fit = wavelengths[lo:hi].astype(np.float64)
    y_fit = spectrum[lo:hi].astype(np.float64)
    if len(x_fit) < 4:
        return None, None, None, None, None
    try:
        amp0 = y_fit.max() - y_fit.min()
        mu0  = wavelengths[peak_idx]
        sig0 = (x_fit[-1] - x_fit[0]) / 6
        off0 = y_fit.min()
        popt, _ = curve_fit(gaussian_fn, x_fit, y_fit,
                            p0=[amp0, mu0, sig0, off0], maxfev=5000)
        amp, mu, sigma, offset = popt
        fwhm = 2.355 * abs(sigma)
        y_pred = gaussian_fn(x_fit, *popt)
        ss_res = np.sum((y_fit - y_pred) ** 2)
        ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-10)
        return amp, mu, sigma, fwhm, r2
    except Exception:
        return None, None, None, None, None


def generate_report(test_type, y_true, y_pred, wavelengths, labels, save_dir):
    """根据测试类型生成并打印报告，同时返回报告文本"""
    os.makedirs(save_dir, exist_ok=True)
    buf = io.StringIO()

    def p(*args, **kwargs):
        print(*args, **kwargs)
        print(*args, **kwargs, file=buf)

    p("\n" + "=" * 60)
    p(f"  🧪 MMI 测试报告 — 类型: {test_type.upper()}")
    p("=" * 60)

    n_samples = len(y_true)
    overall_mse  = np.mean((y_true - y_pred) ** 2)
    overall_mae  = np.mean(np.abs(y_true - y_pred))
    overall_rmse = np.sqrt(overall_mse)
    p(f"\n📊 整体指标  (N={n_samples}):")
    p(f"   MSE = {overall_mse:.6f}    MAE = {overall_mae:.6f}    RMSE = {overall_rmse:.6f}")

    # --- 宽谱报告 ---
    if test_type == "broad":
        p(f"\n📈 宽谱拟合质量:")
        r_vals = [pearsonr(y_true[i], y_pred[i])[0] for i in range(n_samples)]
        avg_r = np.mean(r_vals)
        p(f"   平均 Pearson r = {avg_r:.4f}")
        # 可视化: 随机 4 个样本
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        idxs = np.random.choice(n_samples, min(4, n_samples), replace=False)
        for ax, i in zip(axes.flat, idxs):
            ax.plot(wavelengths, y_true[i], "k-", label="True", linewidth=1.5)
            ax.plot(wavelengths, y_pred[i], "r--", label="Pred", linewidth=1.2)
            ax.set_title(f"Sample: {labels[i]}  (r={r_vals[i]:.4f})")
            ax.legend(fontsize=8)
        fig.tight_layout()
        pth = os.path.join(save_dir, "broad_report.png")
        fig.savefig(pth, dpi=100); plt.close(fig)
        p(f"   📈 对比图: {pth}")

    # --- 单峰报告 ---
    elif test_type == "single":
        p(f"\n🎯 单峰拟合报告 (峰位 + FWHM):")
        p(f"   {'Label':<12s} {'True λ':>9s}  {'Pred λ':>9s}  {'Δλ':>8s}  {'FWHM':>7s}  {'R²':>6s}")
        p(f"   {'-'*60}")
        deltas, fwhms, r2s = [], [], []
        for i in range(n_samples):
            pks, _, pk_idx = detect_peaks_per_sample(
                y_true[i], wavelengths,
                height_rel=cfg.PEAK_HEIGHT_REL, distance=cfg.PEAK_DISTANCE
            )
            if len(pks) == 0:
                continue
            peak_vals = y_true[i][pk_idx]
            best_idx = pk_idx[np.argmax(peak_vals)]
            amp, mu, sigma, fwhm, r2 = _fit_gaussian_to_peak(
                wavelengths, y_pred[i], best_idx, window=cfg.FIT_WINDOW
            )
            if mu is None:
                continue
            true_peak = wavelengths[best_idx]
            delta = mu - true_peak
            deltas.append(abs(delta)); fwhms.append(fwhm); r2s.append(r2)
            p(f"   {labels[i]:<12s} {true_peak:9.3f}  {mu:9.3f}  {delta:+8.4f}  {fwhm:7.3f}  {r2:6.4f}")
        if deltas:
            p(f"   {'-'*60}")
            p(f"   平均:  |Δλ|={np.mean(deltas):.4f} nm   FWHM={np.mean(fwhms):.3f} nm   R²={np.mean(r2s):.4f}")
        # 可视化
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        idxs = np.random.choice(n_samples, min(4, n_samples), replace=False)
        for ax, i in zip(axes.flat, idxs):
            ax.plot(wavelengths, y_true[i], "k-", label="True", linewidth=1.5)
            ax.plot(wavelengths, y_pred[i], "r--", label="Pred", linewidth=1.2)
            ax.set_title(f"Sample: {labels[i]}")
            ax.legend(fontsize=8)
        fig.tight_layout()
        pth = os.path.join(save_dir, "single_report.png")
        fig.savefig(pth, dpi=100); plt.close(fig)
        p(f"   📈 对比图: {pth}")

    # --- 双峰报告 (Resolution) ---
    elif test_type == "dual":
        p(f"\n🔬 双峰分辨率报告:")
        p(f"   {'Label':<12s} {'Peak1_true':>10s} {'Peak1_pred':>10s}  "
          f"{'Peak2_true':>10s} {'Peak2_pred':>10s}  {'True Δλ':>8s} {'Pred Δλ':>8s}  {'Δ error':>8s}")
        p(f"   {'-'*90}")
        delta_errors = []
        for i in range(n_samples):
            pks, _, pk_idx = detect_peaks_per_sample(
                y_true[i], wavelengths,
                height_rel=cfg.PEAK_HEIGHT_REL, distance=cfg.PEAK_DISTANCE
            )
            if len(pks) < 2:
                continue
            idx1, idx2 = int(pk_idx[0]), int(pk_idx[-1])
            true1, true2 = wavelengths[idx1], wavelengths[idx2]
            _, mu1, _, _, _ = _fit_gaussian_to_peak(
                wavelengths, y_pred[i], idx1, window=cfg.FIT_WINDOW
            )
            _, mu2, _, _, _ = _fit_gaussian_to_peak(
                wavelengths, y_pred[i], idx2, window=cfg.FIT_WINDOW
            )
            if mu1 is None or mu2 is None:
                continue
            true_delta = abs(true2 - true1)
            pred_delta = abs(mu2 - mu1)
            err = abs(pred_delta - true_delta)
            delta_errors.append(err)
            p(f"   {labels[i]:<12s} {true1:10.3f} {mu1:10.3f}  "
              f"{true2:10.3f} {mu2:10.3f}  {true_delta:8.3f} {pred_delta:8.3f}  {err:+8.4f}")
        if delta_errors:
            p(f"   {'-'*90}")
            p(f"   平均 Δλ 误差 = {np.mean(delta_errors):.4f} nm   最差 = {np.max(delta_errors):.4f} nm")
        # 可视化
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        idxs = np.random.choice(n_samples, min(4, n_samples), replace=False)
        for ax, i in zip(axes.flat, idxs):
            ax.plot(wavelengths, y_true[i], "k-", label="True", linewidth=1.5)
            ax.plot(wavelengths, y_pred[i], "r--", label="Pred", linewidth=1.2)
            ax.set_title(f"Sample: {labels[i]}")
            ax.legend(fontsize=8)
        fig.tight_layout()
        pth = os.path.join(save_dir, "dual_report.png")
        fig.savefig(pth, dpi=100); plt.close(fig)
        p(f"   📈 对比图: {pth}")

    # --- 多峰报告 ---
    elif test_type == "multi":
        p(f"\n🌈 多峰拟合报告 (各峰位置偏差):")
        # 先确定统一的峰数量
        max_peaks = 0
        for i in range(n_samples):
            pks, _, _ = detect_peaks_per_sample(
                y_true[i], wavelengths,
                height_rel=cfg.PEAK_HEIGHT_REL, distance=cfg.PEAK_DISTANCE
            )
            max_peaks = max(max_peaks, len(pks))
        header = f"   {'Label':<12s}"
        for k in range(max_peaks):
            header += f" {'Peak'+str(k+1)+'_true':>10s} {'Peak'+str(k+1)+'_pred':>10s}"
        header += f"  {'整体MSE':>10s}"
        p(header)
        p(f"   {'-'*len(header)}")
        peak_errors = {k: [] for k in range(max_peaks)}
        for i in range(n_samples):
            pks, _, pk_idx = detect_peaks_per_sample(
                y_true[i], wavelengths,
                height_rel=cfg.PEAK_HEIGHT_REL, distance=cfg.PEAK_DISTANCE
            )
            sample_mse = np.mean((y_true[i] - y_pred[i]) ** 2)
            row = f"   {labels[i]:<12s}"
            for k in range(max_peaks):
                if k < len(pks):
                    idx = int(pk_idx[k])
                    true_l = wavelengths[idx]
                    _, mu, _, _, _ = _fit_gaussian_to_peak(
                        wavelengths, y_pred[i], idx, window=cfg.FIT_WINDOW
                    )
                    pred_l = mu if mu is not None else float("nan")
                    row += f" {true_l:10.3f} {pred_l:10.3f}"
                    if not np.isnan(pred_l):
                        peak_errors[k].append(abs(pred_l - true_l))
                else:
                    row += f" {'---':>10s} {'---':>10s}"
            row += f" {sample_mse:10.6f}"
            p(row)
        p(f"   {'-'*len(header)}")
        for k in range(max_peaks):
            if peak_errors[k]:
                p(f"   Peak{k+1} 平均偏差: ±{np.mean(peak_errors[k]):.4f} nm")
        # 可视化
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        idxs = np.random.choice(n_samples, min(4, n_samples), replace=False)
        for ax, i in zip(axes.flat, idxs):
            ax.plot(wavelengths, y_true[i], "k-", label="True", linewidth=1.5)
            ax.plot(wavelengths, y_pred[i], "r--", label="Pred", linewidth=1.2)
            ax.set_title(f"Sample: {labels[i]}")
            ax.legend(fontsize=8)
        fig.tight_layout()
        pth = os.path.join(save_dir, "multi_report.png")
        fig.savefig(pth, dpi=100); plt.close(fig)
        p(f"   📈 对比图: {pth}")

    # 保存报告文本
    report_path = os.path.join(save_dir, f"test_report_{test_type}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(buf.getvalue())
    p(f"\n📄 报告已保存: {report_path}")
    print("✅ 测试完成!")

    return buf.getvalue()


# ============================================================
# 9. 测试主流程
# ============================================================
def run_testing(model, test_h5_path, save_dir):
    """加载测试数据 → 预测 → 自动分类 → 生成报告"""
    if not os.path.exists(test_h5_path):
        print(f"\n⚠️ 测试文件不存在: {test_h5_path}")
        print("   跳过测试。如需测试，请设置 Config.TEST_H5")
        return

    print("\n" + "=" * 60)
    print("  🧪 开始自动测试")
    print("=" * 60)

    x_test, y_test, wavelengths, labels = load_test_data(test_h5_path)

    print("\n🔮 模型预测...")
    t0 = time.time()
    y_pred = model.predict(x_test, batch_size=cfg.BATCH_SIZE, verbose=1)
    t1 = time.time()
    print(f"   预测完成 ({t1 - t0:.1f}s)  shape={y_pred.shape}")

    if y_pred.ndim == 3:
        y_pred = y_pred.squeeze(-1)

    test_type = auto_detect_test_type(
        y_test, wavelengths,
        height_rel=cfg.PEAK_HEIGHT_REL, distance=cfg.PEAK_DISTANCE
    )
    print(f"   检测到的测试类型: {test_type}")

    generate_report(test_type, y_test, y_pred, wavelengths, labels, save_dir)


# ============================================================
# 10. 主流程
# ============================================================
def main():
    # -- 环境 --
    mount_drive()
    setup_gpu()
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    tf.get_logger().setLevel(logging.ERROR)

    # -- 数据 --
    if not os.path.exists(cfg.TRAIN_H5):
        sys.exit(f"❌ 找不到 {cfg.TRAIN_H5}\n请确认 Google Drive 已挂载且路径正确")
    x_data, y_data = load_data(cfg.TRAIN_H5)
    ds_train, ds_val, train_x, _ = prepare(x_data, y_data)

    input_shape = (train_x.shape[1], train_x.shape[2], train_x.shape[3])
    output_len  = y_data.shape[1] if y_data.ndim == 2 else y_data.shape[-1]
    print(f"\n📐 输入: {input_shape}  输出: {output_len}")

    # -- 模型 --
    gc.collect(); tf.keras.backend.clear_session()
    print("\n🏗️ 构建 ResNet...")
    model = build_resnet_mmi(input_shape, output_len)
    model.summary()
    print(f"参数量: {model.count_params():,}")

    model.compile(
        optimizer=optimizers.Adam(cfg.LEARNING_RATE),
        loss=combined_loss,
        metrics=["mse", "mae"],
    )

    # -- 训练 --
    cbs = [
        EarlyStopping(monitor="val_mse", mode="min", patience=cfg.EARLY_STOPPING_PATIENCE,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(cfg.BEST_MODEL, monitor="val_mse", save_best_only=True,
                        mode="min", verbose=1),
        ReduceLROnPlateau(monitor="val_mse", mode="min", factor=cfg.LR_REDUCE_FACTOR,
                          patience=cfg.LR_REDUCE_PATIENCE, min_lr=cfg.LR_MIN, verbose=1),
        PlotCallback(cfg.SAVE_DIR, cfg.PLOT_FREQUENCY),
    ]

    print(f"\n🚀 开始训练  epochs={cfg.EPOCHS}  batch={cfg.BATCH_SIZE}  lr={cfg.LEARNING_RATE}\n")
    t0 = time.time()
    history = model.fit(ds_train, validation_data=ds_val, epochs=cfg.EPOCHS,
                        verbose=2, callbacks=cbs)
    t1 = time.time()

    # -- 保存 --
    print(f"\n🎉 训练完成! {t1 - t0:.0f}s ({(t1 - t0) / 60:.1f}min)")
    print(f"💾 {cfg.BEST_MODEL}")
    np.savez(os.path.join(cfg.SAVE_DIR, "training_history.npz"),
             loss=history.history.get("loss", []),
             val_loss=history.history.get("val_loss", []),
             mse=history.history.get("mse", []),
             val_mse=history.history.get("val_mse", []),
             mae=history.history.get("mae", []),
             val_mae=history.history.get("val_mae", []))
    print("✅ 训练阶段完成!")

    # -- 自动测试 --
    run_testing(model, cfg.TEST_H5, cfg.SAVE_DIR)
    print("\n✅✅ 全自动训练+测试完成!")


if __name__ == "__main__":
    main()
