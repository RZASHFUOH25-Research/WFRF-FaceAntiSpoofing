# Weighted Frequency Response Filtering for FAS

This repository contains the implementation of the **Weighted Frequency Response Filtering (WFRF)** framework for face anti-spoofing (FAS), as described in the preprint titled "Weighted Frequency Response Filtering for Face Anti-Spoofing" submitted to *Nuclear Physics B* on May 8, 2025. The method reframes domain generalization-based FAS as an anomaly detection problem, using frequency-domain preprocessing and a Manhattan Loss function to improve robustness across diverse domains.

![WFRF Overview](https://raw.githubusercontent.com/Reza-Huddersfield-Student/WFRF-FaceAntiSpoofing/main/images/figure1.jpg)

*Figure 1: Overview of the WFRF preprocessing pipeline, transforming RGB images into the frequency domain, applying a radial weight matrix, and reconstructing enhanced images.*

## Overview

The WFRF framework introduces a novel frequency-based preprocessing technique that amplifies high-frequency components while preserving essential low-frequency cues. By modulating the Fourier spectrum with a radial weight matrix, WFRF enhances discriminative features for spoof detection. The model uses a ResNet-18 backbone and a Manhattan Loss to cluster live samples compactly and separate spoof samples, achieving strong generalization to unseen domains.

**Key contributions**:
- Reframing FAS as an anomaly detection problem in the frequency domain.
- Introducing WFRF to enhance discriminative features via frequency-aware preprocessing.
- Achieving competitive performance in intra- and cross-domain FAS tasks.

## Repository Structure

```
├── data/                    # Directory for datasets (not included)
├── src/                     # Source code
│   ├── preprocessing.py     # WFRF preprocessing implementation
│   ├── model.py             # ResNet-18 backbone and anomaly detection framework
│   ├── train.py             # Training script with Manhattan Loss
│   ├── evaluate.py          # Evaluation script for intra- and cross-domain protocols
│   └── utils.py             # Utility functions (data loading, metrics, etc.)
├── configs/                 # Configuration files
│   └── config.yaml          # Hyperparameters and experiment settings
├── images/                  # Directory for README images
│   ├── figure1.jpg          # Wfrf_pipeline
│   ├── figure4.jpg          # t_SNE visualization
│   ├── figure5.jpg          # Gradcam
│   ├── figure7.jpg          # Convergance
├── results/                 # Directory for model outputs and visualizations
├── requirements.txt         # Python dependencies
└── README.md               # This file
```



1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   Dependencies include:
   - `torch>=1.9.0`
   - `torchvision>=0.10.0`
   - `numpy>=1.19.0`
   - `opencv-python>=4.5.0`
   - `scikit-learn>=0.24.0`
   - `matplotlib>=3.3.0`
   - `pyyaml>=5.4.0`

4. **Hardware requirements**:
   - NVIDIA GPU (e.g., RTX 4060) recommended.
   - At least 32GB RAM and 150GB disk space for datasets.

## Datasets

Experiments use the following datasets:
- **Oulu-NPU**: Intra-domain evaluation.
- **CASIA-FASD**, **Idiap Replay-Attack**, **MSU-MFSD**, **Oulu-NPU**: Cross-domain Protocol 1.
- **Rose-Youtu**, **SiW-Mv2**: Cross-domain Protocol 2.

**Note**: Datasets are not included due to licensing and size constraints. Download them from their respective sources and place them in `data/`. Update `config.yaml` with the correct paths.

## Usage

1. **Configure the experiment**:
   Edit `configs/config.yaml` for dataset paths and hyperparameters:
   ```yaml
   dataset:
     root: "./data/"
     names: ["Oulu-NPU", "CASIA-FASD", "Replay-Attack", "MSU-MFSD"]
   model:
     backbone: "resnet18"
     pretrained: true
   training:
     batch_size: 32
     epochs: 50
     learning_rate: 0.0001
     w_low: 0.5
     w_high: 5.0
     radius: 2.0
     margin: 15.0
   ```

2. **Preprocessing**:
   Apply WFRF preprocessing:
   ```bash
   python src/video-crop.py
   python src/apply_frequency_filter.py
   python src/dataset_loader.py  
   ```

3. **Training**:
   Train the model:
   ```bash
   python src/model.py	
   python src/train.py 
   ```

4. **Evaluation**:
   Evaluate on intra- or cross-domain protocols:
   ```bash
   python src/test.py 
   ```


Results are saved in `results/`.

![t-SNE Visualization](https://raw.githubusercontent.com/Reza-Huddersfield-Student/WFRF-FaceAntiSpoofing/main/images/figure4.jpg)
*Figure 4: t-SNE visualization of feature distributions. Left: Without WFRF, showing overlapping clusters. Right: With WFRF, showing enhanced class separation.*

![Grad-CAM Visualization](https://raw.githubusercontent.com/Reza-Huddersfield-Student/WFRF-FaceAntiSpoofing/main/images/figure5.jpg)
*Figure 5: Grad-CAM visualizations on SiW-Mv2 dataset, highlighting focus on central facial regions for live faces and attack-specific artifacts for spoofs.*


**Note**: This repository is for research purposes only. Ensure compliance with dataset licenses and ethical guidelines.
