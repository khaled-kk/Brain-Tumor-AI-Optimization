# 🧠 Brain Tumor AI: Metaheuristic-Driven Neural Optimization

[![Field: Medical AI](https://img.shields.io/badge/Field-Healthcare--Analytics-red.svg)]()
[![Optimization: GWO--Morphogenetic](https://img.shields.io/badge/Optimization-GWO--Morphogenetic-orange.svg)]()
[![Academic: MSc Data Science](https://img.shields.io/badge/Academic-MSc--Data--Science-blue.svg)]()

## 📌 Project Overview
This project presents an advanced medical image analysis system for automated brain tumor classification from MRI scans. Developed as a **Final Group Project** for the *Machine Learning and Data Visualization* course, the system integrates **Transfer Learning (CNNs)** with **Bio-inspired Metaheuristic Optimization** to achieve superior feature selection and diagnostic accuracy.

---

## 🚀 Run the Implementation
The full end-to-end pipeline—from preprocessing to GWO feature selection—is available as an interactive notebook:
👉 **[Open in Google Colab](https://colab.research.google.com/drive/1hdOHLF4ip_Ua-HOv3TR78uMqOH0YIXCE?usp=sharing)**

---

## 🔬 Core Architecture & Optimization
The system leverages a hybrid approach to medical classification:

### 1. Deep Feature Extraction (Backbones)
*   **Custom CNN:** Baseline model built from scratch.
*   **Transfer Learning Backbones:** VGG16, ResNet50, and DenseNet121 used for robust feature encoding.

### 2. Metaheuristic Feature Selection
Instead of using raw high-dimensional features, this project employs two specialized optimization techniques to reduce noise and improve SVM classification:
*   **Grey Wolf Optimizer (GWO):** Applied to DenseNet121 features to find the optimal diagnostic subset.
*   **Morphogenetic Algorithm:** An experimental bio-inspired method mimicking cellular differentiation for adaptive feature evolution.

---

## 📊 Dataset & Analytics
*   **Domain:** Healthcare — Medical Image Analysis (Brain MRI).
*   **Classes:** Glioma, Meningioma, Pituitary Tumor, and No Tumor.
*   **Dynamic Preprocessing:** Images resized to **224 × 224** with on-the-fly normalization ($1/255$) via Keras `ImageDataGenerator`.

---

## 📈 Evaluation & Visualizations
The project includes deep analytical insights communicated through static visualizations:
*   **Convergence Analysis:** Tracking GWO stability and feature reduction.
*   **Performance Benchmarks:** Accuracy vs. Computational Cost scatter plots.
*   **Error Analysis:** Confusion Matrix heatmaps for multi-class diagnostic precision.

---

## 📂 Repository Structure
```text
├── Final_Report.pdf       # Full academic analysis and results
├── Code.pdf               # Exported execution log from Google Colab
├── Presentation.pptx      # High-level executive summary
├── Images/                # Accuracy plots, loss curves, and GWO results
└── README.md
```
---
*Developed by Khaled Walid*
