# rnn.py

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Dict


class LSTMVariantModel(nn.Module):
    

    def __init__(self, in_channels: int, hidden_size: int, num_layers: int, num_classes: int, dropout: float):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        x shape: (batch, seq_len, in_channels)
        """
        out, _ = self.lstm(x)    # out shape: (batch, seq_len, hidden)
        out = out[:, -1, :]      # last timestep
        out = self.fc(out)
        return out


def build_model(config):
    """
    Build LSTM model from the config dictionary.
    """
    model = LSTMVariantModel(
        in_channels=config["in_channels"],
        hidden_size=64,
        num_layers=2,
        num_classes=config["num_classes"],
        dropout=config["dropout"],
    )
    return model.to(config["device"])


# -----------------------------
# Early Stopping
# -----------------------------

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.should_stop = False

    def step(self, loss):
        if self.best_loss is None:
            self.best_loss = loss
            return

        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.should_stop = True


# -----------------------------
# Training Loop
# -----------------------------

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        preds = model(X)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)

    return total_loss / len(dataloader.dataset)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            preds = model(X)
            loss = criterion(preds, y)
            total_loss += loss.item() * X.size(0)

    return total_loss / len(dataloader.dataset)


def train_model(model, train_loader, val_loader, config):
    """
    Full training loop with early stopping built in.
    """

    device = config["device"]
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )

    early = None
    if config["early_stopping"]["enabled"]:
        early = EarlyStopping(
            patience=config["early_stopping"]["patience"],
            min_delta=config["early_stopping"]["min_delta"]
        )

    for epoch in range(config["epochs"]):

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{config['epochs']} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if early:
            early.step(val_loss)
            if early.should_stop:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    return model
