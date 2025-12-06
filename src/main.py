# main.py

import os
import argparse
import itertools
import numpy as np
import torch
import matplotlib.pyplot as plt

from copy import deepcopy

from config import config, print_config
from rnn import build_model, train_model, evaluate
from data import create_dataloaders, CovidTimeseriesDataset
from utils import set_seed, regression_metrics


CSV_PATH = "owid-covid-data.csv"
COUNTRY = "Denmark"
SEQ_LEN = 30
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "denmark_lstm_best.pth")
PLOTS_DIR = "plots"


# ---------------------------------------------------------------------
# Hyperparameter sweep (small rounds)
# ---------------------------------------------------------------------
def run_sweep(train_loader, val_loader, num_features: int):
    """
    Grid search over dropout, lr, weight_decay using small 'screening_epochs'.
    Returns best hyperparameters and sweep results.
    """
    sweep_cfg = config["sweep"]
    param_grid = sweep_cfg["param_grid"]
    screening_epochs = sweep_cfg["screening_epochs"]

    dropout_vals = param_grid["dropout"]
    lr_vals = param_grid["lr"]
    wd_vals = param_grid["weight_decay"]

    results = []
    best_val_loss = float("inf")
    best_params = None

    print("\n=== Sweep: small-screening runs ===")
    print(f"Dropout: {dropout_vals}")
    print(f"LR: {lr_vals}")
    print(f"Weight decay: {wd_vals}")
    print(f"Screening epochs: {screening_epochs}")
    print("=" * 50)

    for dropout, lr, wd in itertools.product(dropout_vals, lr_vals, wd_vals):
        # copy base config and modify
        candidate_cfg = deepcopy(config)
        candidate_cfg["dropout"] = dropout
        candidate_cfg["lr"] = lr
        candidate_cfg["weight_decay"] = wd
        candidate_cfg["epochs"] = screening_epochs

        # Disable early stopping during screening to make runs more comparable
        candidate_cfg["early_stopping"]["enabled"] = False

        print(f"\n[SWEEP] Testing config: dropout={dropout}, lr={lr}, weight_decay={wd}")

        set_seed(candidate_cfg["seed"])
        model = build_model({
            **candidate_cfg,
            "in_channels": num_features,
            "num_classes": num_features,
        })

        # Small training run
        model = train_model(model, train_loader, val_loader, candidate_cfg)

        # Evaluate on validation set
        criterion = torch.nn.MSELoss()
        val_loss = evaluate(model, val_loader, criterion, candidate_cfg["device"])
        print(f"[SWEEP] Val Loss (MSE): {val_loss:.4f}")

        results.append({
            "dropout": dropout,
            "lr": lr,
            "weight_decay": wd,
            "val_loss": val_loss,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = {"dropout": dropout, "lr": lr, "weight_decay": wd}

    print("\n=== Sweep finished ===")
    print(f"Best config: {best_params}, Val Loss: {best_val_loss:.4f}")
    return best_params, results


# ---------------------------------------------------------------------
# Full training + evaluation + plots + next-step prediction
# ---------------------------------------------------------------------
def full_train_and_evaluate(best_params, train_loader, val_loader, test_loader, num_features, features):
    """
    Train final model with best hyperparameters for full_epochs,
    evaluate on test set, save model, plot results, predict next step.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    full_epochs = config["sweep"]["full_epochs"]
    final_cfg = deepcopy(config)
    final_cfg.update(best_params)
    final_cfg["epochs"] = full_epochs
    # Re-enable early stopping as defined in base config
    final_cfg["early_stopping"] = deepcopy(config["early_stopping"])

    print("\n=== Full training with best hyperparameters ===")
    print(f"Using epochs={full_epochs}, dropout={final_cfg['dropout']}, "
          f"lr={final_cfg['lr']}, weight_decay={final_cfg['weight_decay']}")

    print_config()

    set_seed(final_cfg["seed"])
    model = build_model({
        **final_cfg,
        "in_channels": num_features,
        "num_classes": num_features,
    })

    model = train_model(model, train_loader, val_loader, final_cfg)

    # --- Evaluate on test set ---
    criterion = torch.nn.MSELoss()
    test_loss = evaluate(model, test_loader, criterion, final_cfg["device"])
    print(f"\nFinal Test Loss (MSE): {test_loss:.4f}")

    # Collect predictions on test
    model.eval()
    all_true, all_pred = [], []

    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(final_cfg["device"])
            preds = model(X)
            all_true.append(y.numpy())
            all_pred.append(preds.cpu().numpy())

    all_true = np.concatenate(all_true, axis=0)
    all_pred = np.concatenate(all_pred, axis=0)

    metrics = regression_metrics(all_true, all_pred)
    print(f"Final Test MAE:   {metrics['mae']:.4f}")
    print(f"Final Test RMSE:  {metrics['rmse']:.4f}")
    print(f"Final Test R^2:   {metrics['r2']:.4f}")
    print(f"Final Test Acc@10% (within 10% relative error): {metrics['acc_10'] * 100:.2f}%")

    # --- Save model checkpoint with best params ---
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "features": features,
            "seq_len": SEQ_LEN,
            "country": COUNTRY,
            "best_params": best_params,
        },
        MODEL_PATH,
    )
    print(f"\nSaved best model to {MODEL_PATH}")

    # --- Plot test true vs predicted for each feature ---
    print("\nGenerating plots for test set...")

    # Rebuild full dataset to get dates aligned
    full_ds = CovidTimeseriesDataset(
        csv_path=CSV_PATH,
        country=COUNTRY,
        seq_len=SEQ_LEN,
        features=features,
        scaler=None,
        fit_scaler=True,
    )
    # We must match the same split as create_dataloaders defaults
    n = len(full_ds)
    train_ratio, val_ratio = 0.7, 0.15
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    # test indices in dataset.y / dataset.dates
    test_start = n_train + n_val
    test_dates = full_ds.dates[test_start:]

    if len(test_dates) != all_true.shape[0]:
        print("[Warning] Mismatch between test dates and prediction length; plotting with index instead of dates.")
        x_axis = np.arange(all_true.shape[0])
    else:
        x_axis = test_dates

    for i, feat_name in enumerate(features):
        plt.figure(figsize=(10, 4))
        plt.plot(x_axis, all_true[:, i], label="True")
        plt.plot(x_axis, all_pred[:, i], linestyle="--", label="Predicted")
        plt.title(f"{COUNTRY} - {feat_name} (Test set)")
        plt.xlabel("Time")
        plt.ylabel("Scaled value")
        plt.legend()
        plt.tight_layout()

        out_path = os.path.join(PLOTS_DIR, f"test_{feat_name}.png")
        plt.savefig(out_path)
        plt.close()
        print(f"Saved plot: {out_path}")

    # --- Predict next step after last date ---
    print("\nPredicting next step after last available date...")

    data_array = full_ds.data_array
    last_seq = data_array[-SEQ_LEN:]
    last_seq_t = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).to(final_cfg["device"])

    with torch.no_grad():
        pred_scaled = model(last_seq_t).cpu().numpy()[0]

    # Inverse transform to original scale
    scaler = full_ds.scaler
    pred_original = scaler.inverse_transform(pred_scaled.reshape(1, -1))[0]

    print("\nNext-step prediction (per feature, original scale):")
    for name, value in zip(features, pred_original):
        print(f"  {name}: {value:.4f}")

    # Save a simple bar plot for the next-step forecast
    plt.figure(figsize=(8, 4))
    plt.bar(features, pred_original)
    plt.xticks(rotation=45)
    plt.title(f"{COUNTRY} - Next-step forecast (original scale)")
    plt.tight_layout()
    forecast_path = os.path.join(PLOTS_DIR, "next_step_forecast.png")
    plt.savefig(forecast_path)
    plt.close()
    print(f"Saved next-step forecast plot: {forecast_path}")


# ---------------------------------------------------------------------
# Full Option B pipeline
# ---------------------------------------------------------------------
def run_auto_pipeline():
    
    set_seed(config["seed"])

    device = config["device"]
    print(f"Using device: {device}")
    print_config()

    print("\n[1/3] Creating dataloaders...")
    train_loader, val_loader, test_loader, scaler, features = create_dataloaders(
        csv_path=CSV_PATH,
        country=COUNTRY,
        seq_len=SEQ_LEN,
        batch_size=config["batch_size"],
        features=None,  # uses DEFAULT_FEATURES from data.py
        train_ratio=0.7,
        val_ratio=0.15,
    )
    num_features = len(features)
    print(f"Detected {num_features} features: {features}")

    print("\n[2/3] Running hyperparameter sweep (small rounds)...")
    best_params, sweep_results = run_sweep(train_loader, val_loader, num_features)

    print("\n[3/3] Full training, evaluation, plotting, and next-step prediction...")
    full_train_and_evaluate(best_params, train_loader, val_loader, test_loader, num_features, features)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="COVID LSTM Pipeline with Sweep")
    parser.add_argument(
        "--mode",
        type=str,
        default="auto",
        choices=["auto", "sweep", "train", "predict-only"],
        help=(
            "Mode:\n"
            "  auto          - sweep + full train + eval + plots + next-step prediction (Option B)\n"
            "  sweep         - only run the small screening sweep and print best params\n"
            "  train         - skip sweep, train once with config.py, eval, plots, prediction\n"
            "  predict-only  - load best model and only do next-step prediction\n"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "auto":
        run_auto_pipeline()

    elif args.mode == "sweep":
        # Just run the sweep and print best config
        set_seed(config["seed"])
        train_loader, val_loader, _, _, features = create_dataloaders(
            csv_path=CSV_PATH,
            country=COUNTRY,
            seq_len=SEQ_LEN,
            batch_size=config["batch_size"],
            features=None,
            train_ratio=0.7,
            val_ratio=0.15,
        )
        num_features = len(features)
        run_sweep(train_loader, val_loader, num_features)

    elif args.mode == "train":
        # Use config as-is (no sweep), then full training + plots + prediction
        set_seed(config["seed"])
        train_loader, val_loader, test_loader, scaler, features = create_dataloaders(
            csv_path=CSV_PATH,
            country=COUNTRY,
            seq_len=SEQ_LEN,
            batch_size=config["batch_size"],
            features=None,
            train_ratio=0.7,
            val_ratio=0.15,
        )
        num_features = len(features)
        best_params = {
            "dropout": config["dropout"],
            "lr": config["lr"],
            "weight_decay": config["weight_decay"],
        }
        full_train_and_evaluate(best_params, train_loader, val_loader, test_loader, num_features, features)

    elif args.mode == "predict-only":
        # Load best model and only run next-step prediction
        device = config["device"]
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        features = checkpoint["features"]

        full_ds = CovidTimeseriesDataset(
            csv_path=CSV_PATH,
            country=COUNTRY,
            seq_len=SEQ_LEN,
            features=features,
            scaler=None,
            fit_scaler=True,
        )
        num_features = len(features)

        model = build_model({
            **config,
            "in_channels": num_features,
            "num_classes": num_features,
        })
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        data_array = full_ds.data_array
        last_seq = data_array[-SEQ_LEN:]
        last_seq_t = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_scaled = model(last_seq_t).cpu().numpy()[0]

        scaler = full_ds.scaler
        pred_original = scaler.inverse_transform(pred_scaled.reshape(1, -1))[0]

        print("\nNext-step prediction (per feature, original scale):")
        for name, value in zip(features, pred_original):
            print(f"  {name}: {value:.4f}")


if __name__ == "__main__":
    main()
