# Left Atrium Segmentation from 3D Cardiac MRI

> End-to-end 3D medical image segmentation pipeline for automatic left atrium delineation from cardiac MRI using a 3D Residual UNet.

![Atrium Segmentation Animation](assets/animation.gif)

---

## Results

| Metric | Score |
|---|---|
| Dice Score | 0.80 |
| Hausdorff Distance 95% | 14mm |
| Architecture | 3D Residual UNet |
| Training subjects | 17 |
| Validation subjects | 3 |
| Framework | PyTorch + MONAI |

> Hausdorff Distance reduced from 104mm → 14mm through largest connected component post-processing.

---

## Sample Predictions

| Ground Truth | Prediction |
|---|---|
| ![gt](assets/ground_truth.png) | ![pred](assets/prediction.png) |

*Red overlay = segmented left atrium. Green = ground truth.*

---

## Dataset

- **Source:** [Medical Segmentation Decathlon](http://medicaldecathlon.com) — Task02 Heart
- **Modality:** 3D Cardiac MRI (NIfTI format)
- **Size:** 20 labelled training volumes
- **Task:** Binary segmentation of the left atrium

---

## Pipeline Overview

```
Raw MRI (.nii.gz)
      ↓
Preprocessing (torchio)
  - ToCanonical (RAS+ orientation)
  - CropOrPad (208 × 208 × 128)
  - Z-Normalization
  - Clamp (−3, 3)
  - RescaleIntensity (0, 1)
      ↓
Augmentation (train only)
  - RandomFlip
  - RandomAffine
  - RandomNoise
      ↓
3D Residual UNet (MONAI)
  - 3D convolutions — sees full volume
  - Channels: (16, 32, 64, 128, 256)
  - 4 downsampling levels
  - Skip connections
      ↓
Post-processing
  - Threshold at 0.5
  - Keep Largest Connected Component
      ↓
Evaluation
  - Dice Score
  - Hausdorff Distance 95%
```

---

## Architecture

The model uses MONAI's 3D Residual UNet — a full volumetric architecture that processes the entire 3D MRI volume at once, preserving spatial context across all depth slices.

```
Input (1, 208, 208, 128)
        ↓
  Encoder (↓)          Decoder (↑)
  Conv 1→16    ─────►  Cat + Conv 256+128→128
  Conv 16→32   ─────►  Cat + Conv 128+64→64
  Conv 32→64   ─────►  Cat + Conv 64+32→32
  Conv 64→128  ─────►  Cat + Conv 32+16→16
  Conv 128→256
        ↓
  Output (1, 208, 208, 128) — binary atrium mask
```

Key design choices:
- **3D convolutions** — captures volumetric context (unlike 2D slice-by-slice)
- **Skip connections** — preserves spatial detail lost during downsampling
- **Residual units** — helps gradient flow in deep networks
- **Dice Loss** — handles class imbalance (atrium is ~4% of total volume)

---

## Training Details

| Parameter | Value |
|---|---|
| Loss function | MONAI DiceLoss (sigmoid=True) |
| Optimizer | Adam (lr=1e-4) |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=5) |
| Epochs | 100 |
| Batch size | 1 |
| Precision | 16-bit mixed (fp16) |
| Early stopping | patience=15 |
| Framework | PyTorch Lightning |

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/atrium-segmentation.git
cd atrium-segmentation
pip install -r requirements.txt
```

---

## Requirements

```
torch
torchio
monai
pytorch-lightning
nibabel
scikit-learn
matplotlib
numpy
tqdm
celluloid
```

---

## Usage

**1. Download the dataset**
```
http://medicaldecathlon.com → Task02_Heart
```

**2. Update paths in preprocessing.py**
```python
root  = Path("path/to/Task02_Heart/imagesTr")
label = Path("path/to/Task02_Heart/labelsTr")
```

**3. Run preprocessing and training**
```bash
python train.py
```

**4. Evaluate**
```bash
python evaluate.py --checkpoint path/to/best-model.ckpt
```

---

## Project Structure

```
atrium-segmentation/
├── README.md
├── requirements.txt
├── preprocessing.py      ← torchio pipeline
├── model.py              ← 3D UNet + Lightning module
├── train.py              ← training script
├── evaluate.py           ← metrics + visualization
└── assets/
    ├── animation.gif     ← scrolling prediction GIF
    ├── ground_truth.png  ← sample ground truth
    └── prediction.png    ← sample prediction
```

---

## Limitations

- Only 20 labelled subjects available — limits generalization
- No pretrained encoder weights — trained from scratch
- Single train/val split — cross validation would give more reliable metrics
- Free Colab GPU (T4) — longer training not feasible

## Future Work

- Transfer learning with pretrained encoder (ResNet34/ImageNet)
- Test time augmentation for improved predictions
- 5-fold cross validation for robust evaluation
- Submit to [Medical Decathlon leaderboard](http://medicaldecathlon.com)
- Extend to multi-class cardiac segmentation

---

## References

- [Medical Segmentation Decathlon](http://medicaldecathlon.com)
- [MONAI — Medical Open Network for AI](https://monai.io)
- [torchio — Medical image preprocessing](https://torchio.readthedocs.io)
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) — Ronneberger et al., 2015
- [V-Net: Dice Loss for Medical Segmentation](https://arxiv.org/abs/1606.04797) — Milletari et al., 2016

---

## Author

**Chosen Otabor**
Medical AI Researcher | ALX Backend Engineering
- GitHub: [@YOUR_USERNAME](https://github.com/YOUR_USERNAME)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/YOUR_PROFILE)

---

*Built as part of a medical imaging AI research portfolio.*
