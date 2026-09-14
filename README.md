# Industrial Defect Inspection & Surface Crack Detection via Custom Linear Filtering and Edge Processing

A from-scratch NumPy implementation of a classical computer vision pipeline for industrial defect inspection, covering 2D/3D convolution, Gaussian smoothing, unsharp masking, pooling, and a full 4-stage Canny edge detector built without relying on OpenCV's built-in filtering functions.

## Overview

This project implements each stage of a visual quality inspection pipeline manually using pure NumPy, and benchmarks it on two datasets:

- **PCB Defect Dataset** — high-contrast circuit board images
- **Surface Crack Detection Dataset** — lower-contrast concrete surface images

## Features

- **`conv2d`** — 2D spatial convolution with configurable stride and `same`/`valid` padding
- **`conv3d`** — multi-channel (RGB) convolution producing a stacked activation volume
- **Gaussian kernel generation** — isotropic 2D Gaussian from first principles
- **Unsharp masking** — high-pass detail sharpening
- **`pool2d`** — max/average pooling
- **Canny edge detection** — Sobel gradients → non-maximum suppression → double thresholding → 8-connected hysteresis, all implemented manually
- **Otsu-based adaptive thresholding** — automatically derives edge thresholds per image instead of relying on fixed constants, so the same pipeline works across both high-contrast (PCB) and low-contrast (Surface Crack) datasets
- **Ablation studies** — noise vs. smoothing, before/after NMS, hysteresis vs. single thresholding
- **Scale-space, sensitivity, and failure-case analysis**


## Requirements

```bash
pip install numpy opencv-python matplotlib
```

> `cv2` is used **only** for image I/O (`imread`/`imwrite`) and color conversion (`cvtColor`) — no built-in filtering functions are used.

## Usage

Run the notebook in `src/` top to bottom. Each task section (Task 1–5) is self-contained and prints/plots its own results; final figures are saved automatically to `output_images/`.

## Datasets

- [PCB Defect Dataset](https://www.kaggle.com/datasets/akhatova/pcb-defects)
- [Surface Crack Detection Dataset](https://www.kaggle.com/datasets/arunrk7/surface-crack-detection)

