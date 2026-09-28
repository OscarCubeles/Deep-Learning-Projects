# Import all autoencoder models
from models.simple_conv_autoencoder import simpleconv_autoencoder
from models.deep_conv_autoencoder import deep_conv_autoencoder
from models.simple_sparse_autoencoder import simple_sparse_autoencoder



# Training Configurations for Autoencoders
CONFIGS = {
    'simple_conv': {
        'name': 'Simple Convolutional Autoencoder',
        'model_fn': simpleconv_autoencoder,
        'model_type': 'standard',
        'batch_size': 32,
        'val_split': 0.2,
        'seed': 42,
        'autoencoder': {
            'input_size': (64, 64),
            'learning_rate': 0.001,
            'num_epochs': 50,
        }
    },
    'deep_conv': {
        'name': 'Deep Convolutional Autoencoder',
        'model_fn': deep_conv_autoencoder,
        'model_type': 'standard',
        'batch_size': 32,
        'val_split': 0.2,
        'seed': 42,
        'autoencoder': {
            'input_size': (64, 64),
            'learning_rate': 0.0005,
            'num_epochs': 100,
        }
    },
    'simple_sparse': {
        'name': 'Simple Sparse Autoencoder (L1)',
        'model_fn': simple_sparse_autoencoder,
        'model_type': 'standard',
        'batch_size': 32,
        'val_split': 0.2,
        'seed': 42,
        'autoencoder': {
            'input_size': (64, 64),
            'learning_rate': 0.001,
            'num_epochs': 50,
            'sparsity_weight': 1e-5,
        }
    }
}

