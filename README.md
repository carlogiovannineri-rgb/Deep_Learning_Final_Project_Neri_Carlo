# Skin Cancer Detection using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A multimodal deep learning approach for detecting malignant skin lesions using the ISIC 2024 Challenge dataset (SLICE-3D). This project was developed as part of the **DTSA 5511 - Introduction to Deep Learning** final project.

![ROC Curves](https://img.shields.io/badge/Best%20AUC-0.93-brightgreen)
![pAUC](https://img.shields.io/badge/pAUC%40TPR80-0.15-blue)

---

## Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Model Architectures](#model-architectures)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Key Findings](#key-findings)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [References](#references)

---

## Overview

Skin cancer is one of the most common cancers worldwide, with melanoma being the deadliest form. Early detection significantly improves survival rates, but access to dermatological expertise is limited in many regions.

This project develops and compares multiple deep learning approaches for automated skin lesion classification:

- **Image-based models**: Baseline CNN, EfficientNet-B0
- **Tabular model**: MLP using automatically computed lesion features
- **Multimodal model**: Combining image and tabular features
- **Ensemble methods**: Averaging predictions from multiple models

### The Challenge

The dataset exhibits **extreme class imbalance** (~1:1000 ratio), making this a "needle in a haystack" problem where naive classifiers fail despite high accuracy scores.

---

## Key Results

### Model Performance

| Model | AUC | pAUC@80% TPR |
|-------|-----|--------------|
| **Ensemble (EfficientNet + Tabular)** | **0.9305** | **0.1509** |
| Weighted Ensemble | 0.9290 | 0.1510 |
| Tabular MLP | 0.9107 | 0.1393 |
| Multimodal | 0.9085 | 0.1343 |
| EfficientNet-B0 | 0.8932 | 0.1254 |
| Baseline CNN | 0.6763 | 0.0322 |

### Top Predictive Features

| Rank | Feature | Importance | Clinical Meaning |
|------|---------|------------|------------------|
| 1 | `tbp_lv_H` | 0.068 | Hue (color tone) |
| 2 | `clin_size_long_diam_mm` | 0.049 | Lesion diameter |
| 3 | `tbp_lv_nevi_confidence` | 0.043 | AI nevus score |
| 4 | `tbp_lv_minorAxisMM` | 0.035 | Lesion minor axis |
| 5 | `tbp_lv_deltaLBnorm` | 0.024 | Color contrast |

---

## Dataset

**Source**: [ISIC 2024 Challenge - SLICE-3D](https://www.kaggle.com/competitions/isic-2024-challenge)

| Characteristic | Value |
|----------------|-------|
| Total Images | 401,059 |
| Malignant Cases | 393 (0.098%) |
| Benign Cases | 400,666 (99.902%) |
| Image Source | 3D Total Body Photography (Vectra WB360) |
| Image Type | 15×15mm lesion crops |
| Tabular Features | 55 columns including ABCDE criteria |

### Class Imbalance

```
Benign:    ████████████████████████████████████████ 400,666 (99.9%)
Malignant: █ 393 (0.1%)
```

---

## Methodology

### 1. Data Splitting Strategy

To prevent **data leakage**, we implement patient-level splitting:

```python
# All lesions from the same patient stay together
train_patients, val_patients = train_test_split(
    patient_ids, test_size=0.2, stratify=has_malignant
)
```

This ensures the model learns generalizable cancer features rather than patient-specific characteristics.

### 2. Handling Class Imbalance

We employ a **moderate approach** to avoid the "double punishment" problem:

| Technique | Implementation |
|-----------|----------------|
| Undersampling | 80,000 samples (all malignant + subset benign) |
| Loss Weighting | `pos_weight = 50` (capped to avoid over-correction) |
| Threshold Optimization | Finding optimal operating points |

### 3. Evaluation Metrics

- **AUC-ROC**: Overall discrimination ability
- **pAUC@80% TPR**: Partial AUC above 80% sensitivity (competition metric)
- **Precision/Recall**: At various thresholds for clinical interpretation

---

## Model Architectures

### Baseline CNN
```
Conv2d(3→32) → BatchNorm → ReLU → MaxPool
Conv2d(32→64) → BatchNorm → ReLU → MaxPool
Conv2d(64→128) → BatchNorm → ReLU → MaxPool
Conv2d(128→256) → BatchNorm → ReLU → MaxPool
AdaptiveAvgPool → Dropout(0.5) → FC(256→128) → FC(128→1)
```

### EfficientNet-B0
- Pretrained on ImageNet
- Custom classifier head with dropout

### Tabular MLP
```
FC(20→128) → BatchNorm → ReLU → Dropout(0.3)
FC(128→64) → BatchNorm → ReLU → Dropout(0.3)
FC(64→32) → BatchNorm → ReLU → Dropout(0.2)
FC(32→1)
```

### Multimodal Fusion
```
Image → EfficientNet → 1280 dims ─┐
                                  ├─→ Concat → FC(1312→256→64→1)
Tabular → MLP → 32 dims ──────────┘
```

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- ~10GB disk space for dataset

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/isic-2024-skin-cancer-detection.git
cd isic-2024-skin-cancer-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```txt
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
h5py>=3.9.0
Pillow>=10.0.0
tqdm>=4.65.0
```

---

## Usage

### Option 1: Kaggle 

1. Go to [Kaggle Notebooks](https://www.kaggle.com/code)
2. Create a new notebook
3. Add the ISIC 2024 dataset: `isic-2024-challenge`
4. Upload `isic_2024_skin_cancer_multimodal.ipynb`
5. Enable GPU: Settings → Accelerator → GPU T4 x2
6. Run All

### Option 2: Local Execution

```bash
# Download dataset from Kaggle
kaggle competitions download -c isic-2024-challenge

# Run notebook
jupyter notebook isic_2024_skin_cancer_multimodal.ipynb
```

### Expected Runtime

| Component | Time (GPU) |
|-----------|------------|
| EDA | ~5 min |
| Model Training (4 models) | ~3-4 hours |
| Evaluation & Plots | ~15 min |
| **Total** | **~4 hours** |

---

## Project Structure

```
isic-2024-skin-cancer-detection/
│
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
├── LICENSE                                # MIT License
│
├── notebooks/
│   └── isic_2024_skin_cancer_multimodal.ipynb  # Main notebook
│
├── src/                                   # (Optional) Modular code
│   ├── data/
│   │   └── dataset.py                     # Dataset classes
│   ├── models/
│   │   ├── baseline_cnn.py
│   │   ├── tabular_mlp.py
│   │   └── multimodal.py
│   └── utils/
│       ├── metrics.py                     # pAUC, threshold optimization
│       └── visualization.py
│
└── results/                               # Generated outputs
    ├── figures/
    └── model_checkpoints/
```

---

## Key Findings

### 1. Tabular Features Are Surprisingly Powerful

The Tabular MLP using only metadata features achieved **AUC 0.91**, competitive with image-based deep learning models. This suggests that the automatically computed ABCDE features contain highly discriminative information.

### 2. Ensemble Methods Improve Robustness

Combining EfficientNet and Tabular MLP predictions improved AUC from 0.89/0.91 to **0.93**, demonstrating complementary information capture.

### 3. The Precision-Recall Trade-off

At the optimal F1 threshold (0.61):
- **Precision**: 24%
- **Recall**: 17%
- Interpretation: A dermatologist would review ~4 images to find 1 true cancer

For screening (threshold ~0.05):
- Higher recall but more false positives
- Appropriate when missing cancer is more costly than additional examinations

### 4. Patient-Level Performance Drop

```
Lesion-Level AUC: 0.9305
Patient-Level AUC: 0.7522
```

When aggregating predictions per patient, performance decreases significantly. This has important implications for clinical deployment.

---

## Limitations

1. **Limited malignant samples**: Only 72 malignant lesions in validation, leading to high metric variance

2. **Single train/val split**: Cross-validation would provide more robust estimates but was computationally prohibitive

3. **Simple fusion architecture**: Concatenation-based fusion; attention mechanisms might capture better feature interactions

4. **Subset of data**: Used 80k of 401k images due to computational constraints

5. **No external validation**: Results may not generalize to different imaging devices or populations

---

## 🔮 Future Work

- [ ] Implement K-fold cross-validation for robust metrics
- [ ] Try Vision Transformers (ViT) for image encoding
- [ ] Attention-based multimodal fusion
- [ ] GradCAM visualizations for explainability
- [ ] Full dataset training with curriculum learning
- [ ] External validation on different datasets

---

## References

1. International Skin Imaging Collaboration. *SLICE-3D 2024 Challenge Dataset*. [https://doi.org/10.34970/2024-slice-3d](https://doi.org/10.34970/2024-slice-3d)

2. Rotemberg, V., et al. (2021). *A patient-centric dataset of images and metadata for identifying melanomas using clinical context*. Scientific Data.

3. Tan, M., & Le, Q. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*. ICML.

4. Esteva, A., et al. (2017). *Dermatologist-level classification of skin cancer with deep neural networks*. Nature, 542, 115-118.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- ISIC Archive for providing the dataset
- Kaggle for computational resources
- Course instructors for guidance

---

## Author

**Carlo Neri**

- Course: DTSA 5511 - Introduction to Deep Learning
- Institution: CU Boulder
- Date: December 2025

---

<p align="center">
  <i></i>
</p>
