# MMI End-facet CNN

A ResNet CNN that predicts OSA spectra from MMI end-facet micrographs, with automatic spectral analysis and reporting.

---

## Model Architecture (v2)

| Component | Spec |
|-----------|------|
| Input | (40, 400, 1) grayscale end-facet image |
| First Conv | 7×7, stride=1, 32 channels |
| Residual blocks | 8 blocks (64→64→128→128→256→256→512→512) |
| SE attention | After every residual block |
| 1×1 bottleneck | 128 channels |
| Head | Flatten → Dense 256 → Dropout → Dense 300 (linear) |
| Parameters | ~11M |

### Loss

```
MSE + 0.1 × GradientLoss + 0.05 × SAM
```

- **Gradient Loss** — penalizes first-order differences, directly optimizing spectral shape, peak positions, and FWHM
- **SAM (Spectral Angle Mapper)** — angular similarity between spectra, scale-invariant

---

## Auto-Testing

Runs automatically after training. The average number of detected peaks in ground-truth spectra determines the report type:

| Type | Avg peaks | Report |
|------|----------|--------|
| **broad** | < 0.5 | MSE / MAE / RMSE / Pearson r |
| **single** | 0.5–1.5 | Peak position error + FWHM + R² |
| **dual** | 1.5–2.5 | Peak spacing / resolution comparison |
| **multi** | ≥ 2.5 | Per-peak position error table |

Outputs:
- `test_report_{type}.txt`
- `{type}_report.png` (True vs Pred overlays)

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
