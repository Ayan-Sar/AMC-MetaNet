# AMC-MetaNet: Adaptive Metric Classifier Meta-Network

**Few-Shot Remote Sensing Scene Classification with Adaptive Metric Learning**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🌍 Overview

AMC-MetaNet is a meta-learning framework for **few-shot remote sensing scene classification**. It addresses the challenge of classifying remote sensing images when only a limited number of labeled samples are available per class.

### Key Features

- **Adaptive Metric Learning**: Task-specific distance metrics with attention-based feature reweighting
- **ResNet-12 Backbone**: Robust feature extraction optimized for meta-learning
- **Prototypical Classification**: Efficient few-shot learning with class prototypes
- **Multi-Dataset Support**: Compatible with NWPU-RESISC45, UCMerced, AID, and WHU-RS19

## 📋 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      AMC-MetaNet                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input Images ──► ResNet-12 ──► Adaptive Metric Module      │
│                   Backbone      ├─ Channel Attention        │
│                                 ├─ Task Conditioning        │
│                                 └─ Distance Computation     │
│                        │                                    │
│                        ▼                                    │
│              Prototypical Classifier                        │
│                        │                                    │
│                        ▼                                    │
│                   Predictions                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/AMC-MetaNet.git
cd AMC-MetaNet

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## 📁 Dataset Preparation

Download and extract datasets to the `data/` directory:

```
data/
├── NWPU-RESISC45/
│   ├── airplane/
│   ├── airport/
│   ├── ...
│   └── wetland/
├── UCMerced_LandUse/
│   ├── agricultural/
│   ├── airplane/
│   └── ...
└── AID/
    ├── Airport/
    ├── BareLand/
    └── ...
```

### Supported Datasets

| Dataset | Classes | Images/Class | Image Size |
|---------|---------|--------------|------------|
| NWPU-RESISC45 | 45 | 700 | 256×256 |
| UCMerced | 21 | 100 | 256×256 |
| AID | 30 | 200-420 | 600×600 |
| WHU-RS19 | 19 | 50 | 600×600 |

## 🏋️ Training

### Basic Training

```bash
# Train with default config
python train.py --config configs/config.yaml

# Train on specific dataset
python train.py --dataset NWPU-RESISC45 --n_way 5 --n_shot 5

# Train 1-shot model
python train.py --dataset NWPU-RESISC45 --n_way 5 --n_shot 1
```

### Advanced Options

```bash
python train.py \
    --config configs/config.yaml \
    --dataset NWPU-RESISC45 \
    --n_way 5 \
    --n_shot 5 \
    --epochs 100 \
    --lr 0.001 \
    --backbone resnet12 \
    --save_dir ./checkpoints \
    --log_dir ./logs
```

### Monitor Training

```bash
tensorboard --logdir ./logs/tensorboard
```

## 📊 Evaluation

```bash
# Evaluate 5-way 1-shot
python test.py --checkpoint checkpoints/best_model.pth --n_shot 1

# Evaluate 5-way 5-shot
python test.py --checkpoint checkpoints/best_model.pth --n_shot 5

# Save results to JSON
python test.py --checkpoint checkpoints/best_model.pth --output results.json
```

## 🎯 Demo

Run inference on custom images:

```bash
python demo.py \
    --checkpoint checkpoints/best_model.pth \
    --support_dir ./demo/support \
    --query_dir ./demo/query \
    --output results.png
```

Expected directory structure for demo:
```
demo/
├── support/
│   ├── class1/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── class2/
│       └── ...
└── query/
    ├── test1.jpg
    ├── test2.jpg
    └── ...
```

## 📈 Results

Performance on NWPU-RESISC45 (5-way classification):

| Method | 1-shot | 5-shot |
|--------|--------|--------|
| ProtoNet | 52.3 ± 0.8% | 71.2 ± 0.6% |
| RS-MetaNet | 58.4 ± 0.9% | 75.6 ± 0.7% |
| **AMC-MetaNet** | **62.1 ± 0.8%** | **78.3 ± 0.6%** |

## 📂 Project Structure

```
AMC-MetaNet/
├── configs/
│   └── config.yaml          # Training configuration
├── data/
│   ├── dataset.py           # Dataset classes
│   ├── sampler.py           # Episodic batch sampler
│   └── transforms.py        # Data augmentation
├── models/
│   ├── backbone.py          # ResNet-12, Conv4
│   ├── metric_module.py     # Adaptive metric learning
│   └── amc_metanet.py       # Main model
├── utils/
│   ├── metrics.py           # Evaluation metrics
│   ├── logger.py            # Logging utilities
│   └── helpers.py           # Helper functions
├── train.py                 # Training script
├── test.py                  # Evaluation script
├── demo.py                  # Inference demo
├── requirements.txt
├── LICENSE
└── README.md
```

## ⚙️ Configuration

Key configuration options in `configs/config.yaml`:

```yaml
# Few-shot settings
few_shot:
  train_way: 5        # N-way for training
  train_shot: 5       # K-shot for training
  test_way: 5         # N-way for testing
  test_shot: 1        # K-shot for testing

# Model settings
model:
  backbone: resnet12  # resnet12 or conv4
  feature_dim: 640
  dropout: 0.5

# Adaptive metric settings
metric:
  use_adaptive: true
  attention_dim: 128
  temperature: 1.0
```

## 🔬 Method Details

### Adaptive Metric Module

The core innovation of AMC-MetaNet is the **Adaptive Metric Module**, which learns task-specific distance metrics:

1. **Channel Attention**: Weights feature channels based on task relevance
2. **Task Conditioning**: Generates task-specific parameters from the support set
3. **Learnable Temperature**: Scales the distance-based logits for optimal softmax behavior

### Training Strategy

- **Episodic Training**: Each training iteration samples an N-way K-shot episode
- **Balanced Loss**: Combines classification loss with prototype diversity and feature alignment terms
- **Data Augmentation**: Random cropping, flipping, rotation, and color jittering

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Prototypical Networks](https://arxiv.org/abs/1703.05175) for the foundational few-shot learning approach
- [RS-MetaNet](https://arxiv.org/abs/2009.13364) for insights on remote sensing meta-learning
- [NWPU-RESISC45](http://www.escience.cn/people/JunweiHan/NWPU-RESISC45.html) dataset creators

---

(Under review in ICASSP 2026)
