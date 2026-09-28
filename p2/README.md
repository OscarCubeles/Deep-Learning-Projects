# Deep Learning Project – Traffic Sign Classification & OOD Detection

## Introduction
This project focuses on training several deep learning models for traffic‑sign recognition using the **GTSRB (German Traffic Sign Recognition Benchmark)** dataset.  
Three supervised classifiers are developed: a **VGG**, a **standard CNN**, and a **ResNet** architecture.  
In addition, multiple **autoencoders** are trained for **out‑of‑distribution (OOD) detection**. 
The final pipeline works in two stages:
1. Autoencoders determine whether an input image is in‑distribution or out‑of‑distribution.
2. Images considered in‑distribution are classified using the trained CNN‑based models.

The goal is to build a robust classification system capable of rejecting anomalous inputs while accurately classifying valid traffic signs.

---

## Repository Structure

### `configurations/`
Contains all training configuration files for each model.  
These include hyperparameters such as: number of epochs, learning rate, etc...

---

### `models/`
Includes the TensorFlow/Keras implementations of all architectures used in the project:
- VGG
- Autoencoder variants for OOD detection  
---

### `german_dataset.py`
Handles all dataset‑related operations for the **GTSRB** dataset

---

### Training Scripts – `train_*.py`
All files following the pattern `train_*.py` are responsible for training models.  
Examples include:
- `train_cnn.py`  
- `train_vgg.py`  
- `train_resnet.py`  
- `train_autoencoders.py`  

Each script loads the corresponding configuration, initializes the model, trains it, and saves the resulting weights and logs.

---

### Evaluation Scripts – `evaluate_autoencoders.py`
Used to evaluate trained autoencoder models and the OOD task.

---

### `plots/`
Contains all scripts used to generate the figures included in the report.

---
