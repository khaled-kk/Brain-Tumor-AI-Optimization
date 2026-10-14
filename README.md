# Metaheuristic-Driven Feature Selection for Brain Tumour MRI Classification

Group project for the Machine Learning module of the MSc Data Science at the University of Europe for Applied Sciences (2025-2026). It compares baseline and pre-trained deep networks for 4-class brain tumour MRI classification, and tests two add-ons: Grey Wolf Optimisation (GWO) feature selection with an SVM, and a morphogenesis-inspired label-propagation classifier.

## What it does

- Loads the Kaggle [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (glioma, meningioma, no tumour, pituitary), resized to 224 x 224.
- Trains and evaluates on the held-out test set:
  - a basic fully connected ANN on raw pixels
  - a small custom CNN (2 conv blocks)
  - VGG16, ResNet50 and DenseNet121 (ImageNet weights, frozen-base training, then fine-tuning of the last layers)
- Extracts 256-d features from the fine-tuned DenseNet121, standardises them, runs a binary GWO-style search for a feature subset (RBF-SVM accuracy as fitness), and trains a final SVM on the selected features.
- Plots training curves, a model accuracy comparison chart, normalised confusion matrices (seaborn) and a PCA projection of the DenseNet121 features.
- Separate notebook: frozen ResNet50 features, k-nearest-neighbour similarity graph, and an iterative "local excitation / local inhibition" update of class activations, starting from 15% labelled training samples (weak supervision). Test samples are classified by a k-NN vote over the learned activations.

## Tech stack

Python, TensorFlow / Keras, scikit-learn (SVC, NearestNeighbors, StandardScaler, PCA), NumPy, pandas, Matplotlib, seaborn. Run on Google Colab.

## Repository contents

| File | Description |
|------|-------------|
| `Metaheuristic-Driven-Feature-Selection-in-Deep-Neural-Networks-for-Automated-Brain-Tumour-Detection.ipynb` | Main notebook: ANN, CNN, VGG16, ResNet50, DenseNet121, DenseNet121 + GWO + SVM, evaluation and plots |
| `Biomimetic_Brain_Tumor_MRI_Improved.ipynb` | Morphogenesis-inspired self-organising classifier on ResNet50 features |
| `Metaheuristic-Driven Feature Selection in Deep Neural Networks for Automated Brain Tumor Detection (1).pdf` | Project report (15 pages) |

## Results

Test-set accuracy from the saved outputs of the main notebook (1,311 test images):

| Model | Test accuracy | Test loss |
|-------|---------------|-----------|
| ResNet50 (pre-trained, fine-tuned) | 92.52% | 0.236 |
| VGG16 (pre-trained, fine-tuned) | 92.45% | 0.332 |
| DenseNet121 (pre-trained, fine-tuned) | 92.45% | 0.226 |
| Custom CNN | 81.92% | 0.454 |
| Basic ANN | 62.85% | 1.013 |
| DenseNet121 + GWO + SVM | 29.40% | - |

The GWO + SVM result is evaluated on an 80/20 split of the training-set features (915 samples), not on the test set, and is close to chance for 4 classes. In the notebook the features are extracted from a training generator with `shuffle=True` while the labels come from `generator.classes` (directory order), so features and labels are likely misaligned; this probably explains the low score more than the feature selection itself.

The morphogenesis-inspired notebook reaches 67% accuracy (macro F1 0.66) on its own 20% hold-out split of 1,143 images.

## How to run

1. Open a notebook in Google Colab:
   - [Main notebook](https://colab.research.google.com/github/khaled-kk/brain-tumor-ai-optimization/blob/main/Metaheuristic-Driven-Feature-Selection-in-Deep-Neural-Networks-for-Automated-Brain-Tumour-Detection.ipynb)
   - [Morphogenesis notebook](https://colab.research.google.com/github/khaled-kk/brain-tumor-ai-optimization/blob/main/Biomimetic_Brain_Tumor_MRI_Improved.ipynb)
2. Get the dataset:
   - Main notebook: upload the dataset to Google Drive so that `Training/` and `Testing/` are under `/content/drive/MyDrive/archive` (or change `base_path`).
   - Morphogenesis notebook: upload your own `kaggle.json` API token; the notebook downloads and unzips the dataset.
3. Use a GPU runtime and run all cells.

---
*Developed by Khaled Walid, with Mujtaba Shaikh and Joshuva Jeemon (MSc Data Science, University of Europe for Applied Sciences)*
