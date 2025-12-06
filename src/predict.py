# predict.py

import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from config import config
from rnn import build_model
from data import CovidVariantDataset


CSV_PATH = "owid-covid-data.csv"
COUNTRY = "Denmark"
MODEL_PATH = "models/denmark_lstm.pth"


def load_trained_model(model_path: str, in_channels: int, num_classes: int, device: str):
    checkpoint = torch.load(model_path, map_location=device)
    model = build_model({
        **config,
        "in_channels": in_channels,
        "num_classes": num_classes,
    })
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def predict_next_step():
    device = config["device"]
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    variants = checkpoint["variants"]
    seq_len = checkpoint["seq_len"]

    # Rebuild dataset to get full scaled series and scaler
    dataset = CovidVariantDataset(
        csv_path=CSV_PATH,
        country=COUNTRY,
        seq_len=seq_len,
        variants=variants,
        scaler=None,
        fit_scaler=True,
    )
    scaler = dataset.scaler
    data_array = dataset.data_array  # (T, num_features)

    # Last sequence
    last_seq = data_array[-seq_len:]
    last_seq = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0)  # (1, seq_len, num_features)

    model, _ = load_trained_model(
        MODEL_PATH,
        in_channels=data_array.shape[1],
        num_classes=data_array.shape[1],
        device=device,
    )

    last_seq = last_seq.to(device)
    with torch.no_grad():
        pred_scaled = model(last_seq).cpu().numpy()[0]   # (num_features,)

    # Inverse transform
    pred_original = scaler.inverse_transform(pred_scaled.reshape(1, -1))[0]

    print("Predicted next step (per variant):")
    for v_name, value in zip(variants, pred_original):
        print(f"  {v_name}: {value:.4f}")


if __name__ == "__main__":
    predict_next_step()
