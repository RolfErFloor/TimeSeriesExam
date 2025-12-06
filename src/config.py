import torch

config = {
    # Model
    "in_channels": 1,
    "num_classes": 4,
    #Class names

    # Data
    

    # Training
    "batch_size": 32,
    "epochs": 50,
    "dropout": 0.3,
    "lr": 0.00001,
    "weight_decay": 1e-4,

    "seed": 42,

    # Device
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # Early stopping
    "early_stopping": {
        "enabled": True,
        "patience": 5,
        "min_delta": 0.001,
    },

    # Grid search
    "sweep": {
    "param_grid": {
        "dropout": [0.0, 0.3, 0.5],
        "lr": [1e-4, 1e-3, 1e-2],
        "weight_decay": [0, 1e-4, 5e-4],
    },
    "screening_epochs": 20,   # small runs: 20 epochs
    "screening_threshold": 0.90,
    "full_epochs": 50,        # final best model: 50 epochs
},

}


def print_config():
    print("=" * 50)
    print("Configuration")
    print("=" * 50)
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 50)


if __name__ == "__main__":
    print_config()