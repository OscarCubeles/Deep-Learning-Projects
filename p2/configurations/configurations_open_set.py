# Configurations for open-set experiments (German traffic dataset)
CONFIG = {
    'vgg': {
        'input_size': (64, 64),
        'batch_size': 32,
        'epochs': 200,
        'lr': 1e-4,
        'num_examples': 10,
    },
    'cnn': {
        'input_size': (64, 64),
        'batch_size': 32,
        'epochs': 150,
        'lr': 1e-4,
        'num_examples': 10,
    },
    'resnet': {
        'input_size': (64, 64),
        'batch_size': 32,
        'epochs': 100,
        'lr': 1e-4,
        'num_examples': 10,
    },
    'autoencoder': {
        'input_size': (64, 64),
        'batch_size': 32,
        'epochs': 10,
        'lr': 1e-3,
        'beta': 1.0,
        'rho': 0.05,
    }
}