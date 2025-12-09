# 🎨 Monet Style Transfer with CycleGAN

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://tensorflow.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF.svg)](https://www.kaggle.com/competitions/gan-getting-started)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Transform ordinary photographs into paintings in the style of Claude Monet using CycleGAN

<p align="center">
  <img src="images/sample_transformation.png" alt="Sample Transformation" width="800"/>
</p>

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [References](#references)
- [License](#license)

## 🎯 Overview

This project implements a **CycleGAN** (Cycle-Consistent Generative Adversarial Network) to perform unpaired image-to-image translation, specifically transforming regular photographs into paintings that resemble the impressionist style of **Claude Monet**.

The project was developed as part of the Kaggle competition ["I'm Something of a Painter Myself"](https://www.kaggle.com/competitions/gan-getting-started).

### What is CycleGAN?

CycleGAN is designed for **unpaired image-to-image translation**. Unlike traditional paired translation methods, CycleGAN can learn to transform images between two domains without requiring exact image pairs.

**Key Innovation - Cycle Consistency:**
- If we transform Photo → Monet → Photo, we should get back the original photo
- This constraint helps preserve the content while changing the style
- Formula: `F(G(x)) ≈ x` and `G(F(y)) ≈ y`

## ✨ Features

- **Complete CycleGAN Implementation** with ResNet-based generators and PatchGAN discriminators
- **Instance Normalization** for improved style transfer quality
- **Comprehensive EDA** with color distribution analysis
- **Data Augmentation** pipeline (random jitter, cropping, flipping)
- **Training Visualization** with loss curves and sample outputs
- **Kaggle Submission** ready (generates 7,000+ Monet-style images)

## 🏗️ Architecture

### Model Components

| Component | Architecture | Parameters |
|-----------|-------------|------------|
| Generator G (Photo→Monet) | Encoder-ResNet-Decoder | ~11M |
| Generator F (Monet→Photo) | Encoder-ResNet-Decoder | ~11M |
| Discriminator D_Monet | PatchGAN (70×70) | ~2.7M |
| Discriminator D_Photo | PatchGAN (70×70) | ~2.7M |

### Generator Architecture
```
Input (256×256×3)
    ↓
Encoder: Conv → InstanceNorm → ReLU (×3)
    ↓
ResNet Blocks (×6)
    ↓
Decoder: ConvTranspose → InstanceNorm → ReLU (×3)
    ↓
Output: Conv → Tanh (256×256×3)
```

### Loss Functions

1. **Adversarial Loss**: Binary Cross-Entropy for real/fake classification
2. **Cycle Consistency Loss**: L1 loss ensuring `F(G(x)) ≈ x`
3. **Identity Loss**: Helps preserve colors when input is already in target domain

## 📊 Dataset

The dataset consists of two unpaired image collections:

| Dataset | Images | Size | Format |
|---------|--------|------|--------|
| Monet Paintings | ~300 | 256×256×3 | TFRecord/JPEG |
| Photographs | ~7,038 | 256×256×3 | TFRecord/JPEG |

### Data Characteristics

- **Monet Paintings**: Impressionistic brushstrokes, soft edges, warm tones, emphasis on blue/green (water lilies, landscapes)
- **Photographs**: Sharp details, realistic colors, diverse subjects

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Setup

```bash
# Clone the repository
git clone https://github.com/carlogiovannineri-rgb/Week-5-GANs.git
cd Week-5-GANs

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```txt
tensorflow>=2.10.0
numpy>=1.21.0
matplotlib>=3.5.0
Pillow>=9.0.0
```

## 💻 Usage

### Training

```python
# Run the Jupyter notebook
jupyter notebook monet-dl.ipynb
```

Or execute the training script:

```python
from model import CycleGAN

# Initialize model
cyclegan = CycleGAN(
    img_size=256,
    n_resnet_blocks=6,
    lambda_cycle=10.0,
    lambda_identity=0.5
)

# Train
cyclegan.train(epochs=25, batch_size=1)
```

### Inference

```python
from model import load_generator
import tensorflow as tf

# Load trained generator
generator = load_generator('checkpoints/generator_g.h5')

# Transform image
photo = tf.io.read_file('input_photo.jpg')
photo = tf.image.decode_jpeg(photo, channels=3)
photo = tf.image.resize(photo, [256, 256])
photo = (photo / 127.5) - 1.0

monet_style = generator(tf.expand_dims(photo, 0), training=False)
```

## 📈 Results

### Training Progress

The model was trained for **25 epochs** with the following hyperparameters:

| Parameter | Value |
|-----------|-------|
| Learning Rate | 2e-4 |
| Beta_1 | 0.5 |
| Batch Size | 1 |
| Lambda Cycle | 10.0 |
| Lambda Identity | 0.5 |

### Sample Transformations

| Original Photo | Monet Style | Cycle Reconstruction |
|----------------|-------------|---------------------|
| ![](images/photo1.jpg) | ![](images/monet1.jpg) | ![](images/cycle1.jpg) |
| ![](images/photo2.jpg) | ![](images/monet2.jpg) | ![](images/cycle2.jpg) |

### Loss Curves

<p align="center">
  <img src="images/loss_curves.png" alt="Loss Curves" width="700"/>
</p>

## 📁 Project Structure

```
Week-5-GANs/
├── monet-dl.ipynb          # Main Jupyter notebook
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── images/                 # Sample images and visualizations
│   ├── sample_transformation.png
│   ├── loss_curves.png
│   └── ...
├── checkpoints/            # Saved model weights
│   ├── generator_g.h5
│   ├── generator_f.h5
│   └── ...
└── generated_images/       # Output Monet-style images
```

## 📚 References

1. **CycleGAN Paper**: Zhu, J. Y., Park, T., Isola, P., & Efros, A. A. (2017). *Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks*. [arXiv:1703.10593](https://arxiv.org/abs/1703.10593)

2. **Original GAN Paper**: Goodfellow, I., et al. (2014). *Generative Adversarial Nets*. [NeurIPS](https://papers.nips.cc/paper/5423-generative-adversarial-nets)

3. **Instance Normalization**: Ulyanov, D., Vedaldi, A., & Lempitsky, V. (2016). *Instance Normalization: The Missing Ingredient for Fast Stylization*. [arXiv:1607.08022](https://arxiv.org/abs/1607.08022)

4. **PatchGAN**: Isola, P., et al. (2017). *Image-to-Image Translation with Conditional Adversarial Networks*. [arXiv:1611.07004](https://arxiv.org/abs/1611.07004)

## 🔮 Future Improvements

- [ ] Learning rate scheduling with linear decay
- [ ] Attention mechanisms for better feature extraction
- [ ] Perceptual loss using VGG features
- [ ] Multi-scale discriminators
- [ ] Progressive training for higher resolution

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Carlo Neri**

- GitHub: [@carlogiovannineri-rgb](https://github.com/carlogiovannineri-rgb)
- Kaggle: [Profile](https://www.kaggle.com/)

---

<p align="center">
  Made with ❤️ and TensorFlow
</p>
