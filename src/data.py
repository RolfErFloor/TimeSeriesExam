# data.py

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from typing import List, Tuple, Optional

# Default features to use from owid-covid-data.csv
DEFAULT_FEATURES = [
    "new_cases",
    "new_deaths",
    "icu_patients",
    "hosp_patients",
]


class CovidTimeseriesDataset(Dataset):
    

    def __init__(
        self,
        csv_path: str,
        country: str = "Denmark",
        seq_len: int = 30,
        features: Optional[List[str]] = None,
        scaler: Optional[MinMaxScaler] = None,
        fit_scaler: bool = True,
    ):
        
        self.csv_path = csv_path
        self.country = country
        self.seq_len = seq_len

        df = pd.read_csv(csv_path)
        df_country = df[df["location"] == country].copy()

        if df_country.empty:
            raise ValueError(f"No rows found for country='{country}' in {csv_path}")

        df_country["date"] = pd.to_datetime(df_country["date"])
        df_country = df_country.sort_values("date")

        if features is None:
            features = DEFAULT_FEATURES

        # Check that all requested features exist
        missing = [f for f in features if f not in df_country.columns]
        if missing:
            raise ValueError(f"Missing feature columns in CSV: {missing}")

        self.features = features

        df_feat = df_country[features].copy()
        df_feat = df_feat.fillna(0.0)  # replace NaNs with 0

        # Scaling
        if scaler is None:
            scaler = MinMaxScaler()

        if fit_scaler:
            scaled = scaler.fit_transform(df_feat.values)
        else:
            scaled = scaler.transform(df_feat.values)

        self.scaler = scaler
        self.data_array = scaled.astype(np.float32)

        # Dates aligned with targets (each y is at time t, after seq_len history)
        all_dates = df_country["date"].values
        if len(all_dates) != len(self.data_array):
            raise RuntimeError("Date length and data length mismatch")

        # For each sequence i -> target at i+seq_len
        self.dates = all_dates[self.seq_len:]

        # Build sequences
        X_list = []
        y_list = []

        for i in range(len(self.data_array) - seq_len):
            X_list.append(self.data_array[i:i+seq_len])
            y_list.append(self.data_array[i+seq_len])

        self.X = torch.tensor(np.stack(X_list))      # (N, seq_len, num_features)
        self.y = torch.tensor(np.stack(y_list))      # (N, num_features)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_dataloaders(
    csv_path: str,
    country: str,
    seq_len: int,
    batch_size: int,
    features: Optional[List[str]] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, MinMaxScaler, List[str]]:
    """
    Creates train/val/test DataLoaders for time series data without shuffling the time order.

    Returns
    -------
    train_loader, val_loader, test_loader, scaler, feature_names
    """
    # First dataset fits scaler
    full_dataset = CovidTimeseriesDataset(
        csv_path=csv_path,
        country=country,
        seq_len=seq_len,
        features=features,
        scaler=None,
        fit_scaler=True,
    )

    scaler = full_dataset.scaler
    feature_names = full_dataset.features

    n = len(full_dataset)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    indices = np.arange(n)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train_subset = torch.utils.data.Subset(full_dataset, train_idx)
    val_subset = torch.utils.data.Subset(full_dataset, val_idx)
    test_subset = torch.utils.data.Subset(full_dataset, test_idx)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader, scaler, feature_names
