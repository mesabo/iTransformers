
# iTransformers: Advanced Transformer-Based Learning

---

**Created on:** 01/12/2025  
**Author:** Mesabo  
**Email:** messouaboya17@gmail.com  
**GitHub:** [https://github.com/mesabo](https://github.com/mesabo)  
**University:** Hosei University  
**Department:** Science and Engineering  
**Lab:** YuLab  

---

**iTransformers** is a comprehensive project focused on understanding, building, and applying Transformer architectures for diverse machine learning tasks, including time series forecasting, classification, and more. This project emphasizes a practical approach using a Python-based environment optimized for CPU and GPU.

---

## Table of Contents
1. [Overview](#overview)
2. [Project Roadmap](#project-roadmap)
3. [Environment Setup](#environment-setup)
4. [Directory Structure](#directory-structure)
5. [Getting Started](#getting-started)
6. [Key Objectives](#key-objectives)

---

## Overview

**iTransformers** is designed to:
- Teach the fundamentals of Transformer models.
- Enable step-by-step implementation of Transformers from scratch.
- Transition Transformer-based architectures to real-world datasets like **multivariate time series**.

The project is built for CPU-based systems to ensure compatibility and efficiency during training and experimentation.

---

## Project Roadmap

### 1. Foundation: Transformer Fundamentals
- Self-Attention and Scaled Dot-Product Attention.
- Multi-Head Attention.
- Positional Encoding.
- Feedforward Networks.

### 2. Implementing Transformer Components
- Scaled Dot-Product Attention Layer.
- Multi-Head Attention Layer.
- Positional Encoding Layer.
- Transformer Encoder and Decoder.

### 3. Assembling the Complete Transformer
- Combining Encoder and Decoder into a full Transformer.
- Implementing padding and look-ahead masks.

### 4. Lightweight Training
- Using CPU-friendly lightweight datasets.
- Training on classification and sequence-to-sequence tasks.

### 5. Advanced Topics
- Fine-tuning pre-trained Transformer models.
- Optimizing Transformer models for time series forecasting.

---

## Environment Setup

To set up the environment, we use **Conda** for managing dependencies. The environment is tailored for CPU-only training.

### Conda Environment Configuration

Save the following environment file as `abe_env.yml`:

```yaml
name: itransformers
channels:
  - conda-forge
  - pytorch
  - defaults
dependencies:
  - python=3.13.1
  - pip
  - numpy
  - pandas
  - scikit-learn
  - scipy
  - matplotlib
  - seaborn
  - tqdm
  - pillow
  - h5py
  - pyyaml
  - pytorch=2.5.1
  - torchvision
  - pip:
      - opacus
      - pytorch-memlab
      - tensorboard
      - torch-tb-profiler
```

### Install Environment

Run the following commands to create and activate the environment:
```bash
conda env create -f itransformers_env.yml
conda activate itransformers
python -c "import torch; print(torch.__version__)"
```

---

## Directory Structure

```
iTransformers/src/
    ├── foundation/
    │   ├── self_attention.py      # Scaled Dot-Product Attention
    │   ├── multi_head_attention.py # Multi-Head Attention
    │   ├── positional_encoding.py # Positional Encoding
    │   ├── transformer_encoder.py # Transformer Encoder
    │   ├── transformer_decoder.py # Transformer Decoder
    │   └── transformer_model.py   # Full Transformer
    ├── utils/
    │   ├── __init__.py            # Makes this a package
    │   ├── data_utils.py          # Utilities for dataset preparation
    │   ├── training_utils.py      # Utilities for training and evaluation
    │   ├── visualization_utils.py # Utilities for visualizing results
    │   └── misc.py                # Miscellaneous utilities
    ├── training/
    │   ├── dataset_preparation.py # Dataset preparation scripts
    │   ├── train_transformer.py   # Training scripts
    ├── advanced/
    │   ├── fine_tuning.py         # Fine-tune pre-trained Transformers
    │   └── time_series_transformer.py # Multivariate time series Transformer
├── examples/
│   ├── seq2seq_example.py     # Sequence-to-sequence task
│   ├── classification_example.py # Classification task
│   └── time_series_example.py # Time series forecasting
├── README.md                  # Documentation (this file)
└── itransformers_env.yml      # Environment file
```

---

## Getting Started

### Step 1: Clone the Repository
```bash
git clone <repository_url>
cd iTransformers
```

### Step 2: Set Up the Environment
```bash
conda env create -f itransformers_env.yml
conda activate itransformers
python -c "import torch; print(torch.__version__)"
```

### Step 3: Run Examples
Navigate to the `examples/` folder to run various examples:
```bash
python seq2seq_example.py
```

---

## Key Objectives
1. Understand the internal mechanics of Transformers.
2. Implement and train Transformer architectures.
3. Apply Transformers to real-world datasets like **multivariate time series**.
4. Explore optimization and fine-tuning techniques.

---