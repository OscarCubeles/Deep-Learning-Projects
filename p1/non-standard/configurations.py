# Non-standard configurations (for InceptionV3, ViTModel)
configurations_non_standard = [
    {
        "scenario": "overfitting",
        "batch_size": 256,
        "epochs": 50, 
        "learning_rate": 5e-2,
        "lr_step": 10,
        "lr_decay": 1.0,
        "momentum": 0.9
    },
    {
        "scenario": "underfitting",
        "batch_size": 16,
        "epochs": 25,
        "learning_rate": 1e-6,
        "lr_step": 50,
        "lr_decay": 1.0,
        "momentum": 0.0
    },
    {
        "scenario": "balanced",
        "batch_size": 32,
        "epochs": 100,
        "learning_rate": 5e-4,
        "lr_step": 15,
        "lr_decay": 0.85,
        "momentum": 0.9
    }
]

configurations_non_standard_vit = [
    {
        "scenario": "overfitting",
        "batch_size": 256,
        "epochs": 50,
        "learning_rate": 1e-3,
        "lr_step": 10,
        "lr_decay": 1.0
    },
    {
        "scenario": "underfitting",
        "batch_size": 256,
        "epochs": 120,
        "learning_rate": 1e-4,
        "lr_step": 10,
        "lr_decay": 0.1
    },
    {
        "scenario": "balanced",
        "batch_size": 64,
        "epochs": 75,
        "learning_rate": 5e-4,
        "lr_step": 20,
        "lr_decay": 0.7
    }
]