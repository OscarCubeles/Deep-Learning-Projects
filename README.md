# Deep Learning Projects

This repository contains the coursework completed for the **Deep Learning** subject. It brings together two projects covering artwork classification, traffic-sign recognition, and out-of-distribution detection using TensorFlow and Keras.

## Projects

1. **Project 1 - Artwork classification:** Compares standard and non-standard deep-learning architectures for classifying artwork from the MAMe dataset into 29 medium categories. The experiments evaluate AlexNet, a custom CNN, ResNet18, InceptionV3, and a Vision Transformer under balanced, underfitting, and overfitting configurations. See the [`project documentation`](p1/README.md), [`source code`](p1), [`experiment results`](p1/results), and [`report`](p1/report.pdf).

2. **Project 2 - Traffic-sign classification and open-set recognition:** Develops CNN, VGG, and ResNet classifiers for German traffic signs and uses convolutional autoencoders to detect out-of-distribution images through reconstruction error. The final pipeline rejects anomalous inputs before classifying recognized traffic signs. See the [`project documentation`](p2/README.md), [`source code`](p2), and [`report`](p2/DL__Project_2_open_set_recognition.pdf).

## Repository Structure

```text
.
├── p1/                         # MAMe artwork classification
│   ├── standard/                # AlexNet, custom CNN, and ResNet18 experiments
│   ├── non-standard/            # InceptionV3 and Vision Transformer experiments
│   ├── results/                 # Training metrics and plots
│   └── report.pdf               # Project report
└── p2/                         # Traffic signs and open-set recognition
    ├── configurations/          # Classifier and autoencoder settings
    ├── models/                  # Autoencoder and evaluation implementations
    ├── plots/                   # Result-visualization scripts
    └── DL__Project_2_open_set_recognition.pdf
```

Each project is independent and includes its own data-loading, training, evaluation, and experiment-configuration code. Dataset and output paths in the scripts must be updated for the environment where the experiments are run.
