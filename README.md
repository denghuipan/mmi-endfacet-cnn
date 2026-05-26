# MMI End-facet CNN — 全自动训练 + 智能测试

从 MMI 端面显微图像预测 OSA 光谱的 ResNet CNN，支持 Colab 一键训练 + 自动测试报告。

---

## Features

- 🚀 **Colab 一键运行** — 4 个 cell，点「全部运行」即可
- 🧠 **ResNet v2** — stride=1 首层 + SE 通道注意力 + Linear 输出
- 📉 **组合损失** — MSE + Gradient Loss + SAM，直接优化光谱形状
- 🧪 **智能测试引擎** — 自动识别 broad / single / dual / multi 四种光谱类型，生成对应报告
- ⚙️ **Notebook 内配置路径** — 不用改 .py 文件

---

## Quick Start (Colab)

1. 打开 `train_colab.ipynb`，连接 Colab GPU
2. 在 Cell 2 修改你的数据目录：

```python
os.environ['MMI_DS_DIR'] = "/content/drive/MyDrive/ucsc/MMI/8um"
```

3. 确保 Google Drive 中有：

```
MyDrive/ucsc/MMI/8um/
├── train.h5                # 训练数据 (augment/x, augment/y)
└── single_test.h5          # 测试数据 (img, osa, wavelength, labels)
```

4. 点「全部运行」→ 训练 → 自动测试 → 报告

---

## Project Structure

```
├── train_colab.ipynb          # Colab 启动器（用户入口）
├── colab_auto.py              # 全自动训练 + 测试脚本
├── model.py                   # ResNet v2 模型定义 + 损失函数
├── auto_train.py              # 本地训练脚本（命令行）
├── config.py                  # 本地训练配置文件
├── utils.py                   # HDF5 数据加载工具
├── requirements.txt           # Python 依赖
├── .gitignore
└── multi peak End-facet CNN.ipynb  # 原始实验 notebook
```

---

## Configuration

### Colab（推荐）

在 `train_colab.ipynb` 的 Cell 2 中设置环境变量：

```python
os.environ['MMI_DS_DIR']   = "/content/drive/MyDrive/ucsc/MMI/8um"
os.environ['MMI_TRAIN_H5'] = "train.h5"
os.environ['MMI_TEST_H5']  = "single_test.h5"
```

### 本地训练

编辑 `config.py`，然后运行：

```bash
python auto_train.py
```

或带参数：

```bash
python auto_train.py --epochs 500 --batch-size 32
```

---

## Model Architecture (v2)

| 组件 | 配置 |
|------|------|
| 输入 | (40, 400, 1) 灰度端面图像 |
| 首层 Conv | 7×7, stride=1, 32ch |
| 残差块 | 8 个 (64→64→128→128→256→256→512→512) |
| SE 注意力 | 每层残差块后 |
| 1×1 降维 | 128ch |
| Dense | 256 → Dropout → 300 (linear) |
| 参数量 | ~11M |

### Loss

```python
Loss = MSE + 0.1 × GradientLoss + 0.05 × SAM
```

- **Gradient Loss**: 一阶差分，惩罚光谱形状差异
- **SAM (Spectral Angle Mapper)**: 光谱角度相似度

可在 Config 中调整权重（设为 0 关闭）：

```python
LOSS_W_GRAD = 0.1
LOSS_W_SAM  = 0.05
```

---

## Auto-Testing

训练完成后自动运行。根据 ground truth 光谱的峰值数量判断类型：

| 类型 | 平均峰值数 | 报告内容 |
|------|-----------|---------|
| **broad** | < 0.5 | MSE / MAE / RMSE / Pearson r |
| **single** | 0.5–1.5 | 峰位偏差 + FWHM + R² |
| **dual** | 1.5–2.5 | 双峰间距 / 分辨率对比 |
| **multi** | ≥ 2.5 | 各峰位置偏差表 |

报告输出：
- `test_report_{type}.txt` — 文本报告
- `{type}_report.png` — True vs Pred 对比图

---

## Data Format

### 训练 H5

```
├── augment/x    (N, H, W)      图像
└── augment/y    (N, L)         光谱
```

### 测试 H5

```
├── img          (N, H, W)      图像
├── osa          (N, L)         光谱
├── wavelength   (L,)           波长数组
└── labels       (N,)           样本标签
```

---

## Requirements

```
tensorflow>=2.13.0
numpy>=1.24.0
scipy>=1.10.0
h5py>=3.8.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
```
