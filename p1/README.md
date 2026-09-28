# Deep Learning CNN Classification Project

This project implements and compares various CNN architectures for image classification on the MAMe dataset. The project explores different model architectures and training scenarios (balanced, overfitting, underfitting) to analyze model performance.

## Project Structure

The project is organized into two main folders: **standard** and **non-standard**, each containing experiments for different CNN architectures.

```
ProjectDL/
├── data_loader.py              # Custom dataset loader for MAMe dataset
├── utils.py                    # General utilities (path definitions, dataset loading)
│
├── standard/                   # Standard architecture experiments
│   ├── launcher_standard.sh    # Launcher script 
│   ├── run_standard_architectures.py # Main runner - executes all experiments
│   ├── standard_models.py      # Architecture definitions (AlexNet, etc.)
│   ├── train_standard_models.py # Training logic for standard models
│   ├── configurations.py       # Hyperparameter configurations per scenario
│   └── figures.py              # Figure generation for report
│
└── non-standard/               # Non-standard architecture experiments
    ├── launcher_non_standard.sh # Launcher script 
    ├── run_non_standard_architectures.py # Main runner - executes all experiments
    ├── non_standard_models.py  # Architecture definitions (InceptionV3, ViT)
    ├── train_non_standard_models.py # Training logic for non-standard models
    ├── configurations.py       # Hyperparameter configurations per scenario
    └── plots.py                # Plot generation for report
```

### Folder Organization

**Root Directory:**
- `data_loader.py`: Implements `MAMeDataset` class for loading and preprocessing images
- `utils.py`: Contains general utilities like dataset paths and configuration variables

**standard/ and non-standard/ Folders:**

Each folder follows the same structure and workflow:

1. **Launcher** (`.sh` files): Bash scripts that submit jobs to the BSC account and execute the respective `run_*_architectures.py` script
2. **Runner** (`run_*_architectures.py`): Orchestrates experiments by iterating through models and configurations
3. **Architectures** (`*_models.py`): Contains the neural network architecture definitions
4. **Training** (`train_*_models.py`): Implements the training loop and evaluation logic
5. **Configurations** (`configurations.py`): Defines hyperparameters for each training scenario
6. **Visualization** (`figures.py` / `plots.py`): Generates plots and figures for the final report

## Implemented Architectures

### Standard Architectures (`standard/standard_models.py`)
- **AlexNet**: Classic CNN architecture adapted for 256×256 images and 29 classes
- **StandardArchitectureModelNoAttention**: Same as above but without attention mechanism
- **ResNet18**: Manually implemented ResNet18 matching torchvision's structure

### Non-Standard Architectures (`non-standard/non_standard_models.py`)
- **InceptionV3**: Transfer learning with custom classification head
- **ViTModel**: Vision Transformer implementation with patch-based processing

## Quick Start

### Step-by-Step Execution

#### 1. Configure Paths
Update the paths in `utils.py` to point to your dataset location:
```python
base_path = '/your/data/path'
csv_path = base_path + '/MAMe_metadata/MAMe_dataset.csv'
img_dir = base_path + '/MAMe_data_256'
```

#### 2. Run Standard Architecture Experiments
```bash
sbatch -A nct_350 -q acc_training launcher_non_standard.sh

```



## Training Scenarios

Each model is trained under three scenarios to study different learning behaviors:

### Standard Models (AlexNet, StandardArchitectureModel)
- **Balanced**: Optimal hyperparameters for best performance (32 batch, 100 epochs, lr=9e-4)
- **Underfitting**: Conservative settings to limit learning (64 batch, 60 epochs, lr=1.2e-5)
- **Overfitting**: Aggressive settings to maximize memorization (16 batch, 110 epochs, lr=1e-3)

### Non-Standard Models (InceptionV3, ViTModel)
- **Balanced**: Optimized for convergence and generalization
- **Underfitting**: Minimal learning configuration
- **Overfitting**: High-capacity training setup

## Results

Results are saved in CSV format with the following structure:
```
results/
├── AlexNet/
│   ├── balanced.csv
│   ├── underfitting.csv
│   └── overfitting.csv
├─ StandardArchitectureModel/
│   ├── balanced.csv
│   ├── underfitting.csv
│   └── overfitting.csv
├───InceptionV3/
│   └── ...
└── ViTModel/
    └── ...
```

Each CSV contains training and validation metrics (loss, accuracy) per epoch.


## Dataset

This project uses the MAMe dataset for classifying artworks by medium (29 classes). The dataset should be organized with:
- `MAMe_dataset.csv`: Metadata with columns (Image file, Medium, Subset)
- `MAMe_data_256/`: Directory containing 256×256 artwork images



