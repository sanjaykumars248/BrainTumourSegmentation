# 🧠 ResNet-UNet for Brain Tumour Segmentation

> A deep learning model combining ResNet and U-Net to segment brain tumours from MRI scans with high accuracy.

## 📄 Overview

This repository implements a hybrid architecture that uses a **ResNet encoder** within a **U-Net decoder framework** to perform semantic segmentation of brain tumours in MRI images.

**"ResNet Encoder Based UNet for Brain Tumour Segmentation"**  
*2025 International Conference on Data Science and Business Systems (ICDSBS)*  
DOI: [10.1109/ICDSBS63635.2025.11031994](https://doi.org/10.1109/ICDSBS63635.2025.11031994)

## 🎯 Features

- ResNet-504 encoder backbone
- U-Net style upsampling and skip connections
- Trained on BraTS2020 datasets
- Dice score: ~0.82
- Includes preprocessing and evaluation scripts

## 🛠️ Tech Stack

- Python 3.9+
- Tensoflow
- NumPy, OpenCV, Matplotlib
- Jupyter / Colab compatible / Kaggle (Coding Playground)

## 🚀 Getting Started

### Clone the repository

#### Data
Processed Data for this Brain Tumour Segmentation is available on Kaggle
 https://kaggle.com/datasets/2326e66ceafad3bc2edc9bb4757eaf5edc9dccfee6be2d7b99470e3a56efea38

```bash
git clone https://github.com/yourusername/resnet-unet-brain-segmentation.git
cd resnet-unet-brain-segmentation
```

Install dependencies
```bash
pip install -r requirements.txt
```

### 🧠 Training
Run the interactive Python notebook *BrainTS_Training.ipynb* in Google Colab or Kaggle Coding Playground.
Download the trained model and  stored in the folder **BestModels/**

### 📊 Evaluation
```
streamlit run app.py
```



