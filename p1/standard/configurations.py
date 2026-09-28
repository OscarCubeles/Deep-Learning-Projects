# Training configurations for all experiments

# Standard configurations (for AlexNet, StandardArchitectureModel)
configurations_standard = [
    {
        "scenario": "balanced",
        "batch_size": 32,
        "epochs": 100,
        "learning_rate": 9e-4,
        "lr_step": 7,
        "lr_decay": 0.68,
        "optimizer_betas": (0.91, 0.991)
    },
    {
        "scenario": "underfitting",
        "batch_size": 64,
        "epochs": 60,
        "learning_rate": 1.2e-5,
        "lr_step": 5,
        "lr_decay": 0.58,
        "optimizer_betas": (0.988, 0.9988)
    },
    {
        "scenario": "overfitting",
        "batch_size": 16,
        "epochs": 110,
        "learning_rate": 1e-3,
        "lr_step": 30,
        "lr_decay": 0.95,
        "optimizer_betas": (0.9, 0.999)
    }
]

