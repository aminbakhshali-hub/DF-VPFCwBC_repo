# Deep-Feature Vector Picture Fuzzy Clustering with Bias Correction (DF-VPFCwBC)

This repository contains the implementation of the DF-VPFCwBC algorithm proposed in the paper  
**"A Deep-Feature Vector Picture Fuzzy Clustering with Bias Correction for 3D Brain MR Image Segmentation"**.

## Overview
DF-VPFCwBC integrates deep semantic feature extraction (via Attention U-Net) with a vector picture fuzzy clustering model that simultaneously performs bias estimation and correction for robust 3D brain MRI segmentation.

## Repository Structure
- `models.py`: PyTorch implementation of Attention U-Net with attention gates.
- `preprocessing.py`: MRI normalization and skull stripping functions.
- `train_unet.py`: Training pipeline for Attention U-Net.
- `extract_features.py`: Extracts deep features from intermediate layers.
- `clustering.py`: Implementation of Vector Picture Fuzzy Clustering with Bias Correction.
- `run_clustering.py`: Main driver script to run clustering.
- `evaluate.py`: Computes segmentation metrics (Dice, Jaccard, PSNR, SSIM, RMSE).
- `config.yaml`: Example configuration file for paths and parameters.
- `requirements.txt`: Dependencies.
- `LICENSE`: MIT license.
- `generate_synthetic_data.py`: Script to create synthetic test images.

## Quick Start
```bash
pip install -r requirements.txt
python generate_synthetic_data.py
python train_unet.py
python extract_features.py
python run_clustering.py
python evaluate.py
