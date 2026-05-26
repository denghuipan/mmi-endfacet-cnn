# MMI End-facet CNN — Auto-Training + Smart Testing

A ResNet CNN that predicts OSA spectra from MMI end-facet micrographs. One-click Colab training with automatic test reporting.

---

## Features

- 🚀 **One-click Colab** — 4 cells, hit "Run All"
- 🧠 **ResNet v2** — stride=1 first conv + SE channel attention + linear output
- 📉 **Combined loss** — MSE + Gradient Loss + SAM, directly optimizes spectral shape
- 🧪 **Smart testing engine** — auto-detects broad / single / dual / multi spectra, generates type-specific reports
- ⚙️ **Notebook-level config** — set paths in a notebook cell; no need to edit .py files

---

## Quick Start (Colab)

1. Open `train_colab.ipynb` and connect to a Colab GPU
2. Edit Cell 2 to point to your data directory:

```python
os.environ['MMI_DS_DIR'] = "/content/drive/MyDrive/ucsc/MMI/8um"
```

3. Make sure your Google Drive contains:

```
MyDrive/ucsc/MMI/8um/
├── train.h5                # training data (augment/x, augment/y)
└── single_test.h5          # test data (img, osa, wavelength, labels)
```

4. Click "Run All" → training → auto-testing → report

---

## Project Structure

```
├── train_colab.ipynb              # Colab launcher (user entry point)
├── colab_auto.py                  # Auto training + testing script
├── model.py                       # ResNet v2 model + loss functions
├── auto_train.py                  # Local training script (CLI)
├── config.py                      # Local training config
├── utils.py                       # HDF5 data loading utilities
├── requirements.txt               # Python dependencies
├── .gitignore
└── multi peak End-facet CNN.ipynb # Original experiment notebook
```

---

## Configuration

### Colab (recommended)

Set environment variables in Cell 2 of `train_colab.ipynb`:

```python
os.environ['MMI_DS_DIR']   = "/content/drive/MyDrive/ucsc/MMI/8um"
os.environ['MMI_TRAIN_H5'] = "train.h5"
os.environ['MMI_TEST_H5']  = "single_test.h5"
```

All other settings (batch size, epochs, learning rate, loss weights) are in the `Config` class inside `colab_auto.py`.

### Local training

Edit `config.py`, then run:

```bash
python auto_train.py
```

Or with overrides:

```bash
python auto_train.py --epochs 500 --batch-size 32
```

---

## Model Architecture (v2)

| Component | Spec |
|-----------|------|
| Input | (40, 400, 1) grayscale end-facet image |
| First Conv | 7×7, stride=1, 32 channels |
| Residual blocks | 8 blocks (64→64→128→128→256→256→512→512) |
| SE attention | After every residual block |
| 1×1 bottleneck | 128 channels |
| Dense head | 256 → Dropout → 300 (linear activation) |
| Parameters | ~11M |

### Loss

```python
Loss = MSE + 0.1 × GradientLoss + 0.05 × SAM
```

- **Gradient Loss**: first-order difference penalty — directly optimizes spectral shape, peak positions, and FWHM
- **SAM (Spectral Angle Mapper)**: angular similarity between spectra, scale-invariant

Weights are adjustable in Config (set to 0 to disable):

```python
LOSS_W_GRAD = 0.1
LOSS_W_SAM  = 0.05
```

---

## Auto-Testing

Runs automatically after training. Sample type is determined by the average number of peaks in ground-truth spectra:

| Type | Avg peaks | Report contents |
|------|----------|-----------------|
| **broad** | < 0.5 | MSE / MAE / RMSE / Pearson r |
| **single** | 0.5–1.5 | Peak position error + FWHM + R² |
| **dual** | 1.5–2.5 | Peak spacing / resolution comparison |
| **multi** | ≥ 2.5 | Per-peak position error table |

Outputs (saved to the checkpoint directory):
- `test_report_{type}.txt` — text report
- `{type}_report.png` — True vs Pred overlay plots

---

## Data Format

### Training H5

```
├── augment/x    (N, H, W)      images
└── augment/y    (N, L)         spectra
```

### Test H5

```
├── img          (N, H, W)      images
├── osa          (N, L)         spectra
├── wavelength   (L,)           wavelength array
└── labels       (N,)           sample labels (strings)
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
