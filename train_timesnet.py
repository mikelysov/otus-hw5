import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

LOOKBACK = 200
PREDICT_STEPS = 1
BATCH_SIZE = 64
LEARNING_RATE = 0.001
MAX_EPOCHS = 100
PATIENCE = 15
VAL_RATIO = 0.15
D_MODEL = 64
N_LAYERS = 3
DROPOUT = 0.1


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = out_channels // 4

        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=1),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(),
        )

        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=1),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(),
            nn.Conv1d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(),
        )

        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=1),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(),
            nn.Conv1d(mid_channels, mid_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, mid_channels, kernel_size=1),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return torch.cat([b1, b2, b3, b4], dim=1)


class TimesBlock(nn.Module):
    def __init__(self, d_model, n_layers=3, dropout=0.1):
        super().__init__()
        self.inception_blocks = nn.ModuleList(
            [InceptionBlock(d_model, d_model) for _ in range(n_layers)]
        )
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.transpose(1, 2)
        for block in self.inception_blocks:
            x = block(x)
        x = self.conv(x)
        x = self.dropout(x)
        return x.transpose(1, 2)


class TimesNet(nn.Module):
    def __init__(self, n_features, d_model=64, n_layers=3, predict_len=1, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.times_block = TimesBlock(d_model, n_layers, dropout)

        self.temporal_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)

        self.grn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

        self.output = nn.Linear(d_model, predict_len)

    def forward(self, x):
        x = self.input_proj(x)

        x = self.times_block(x)

        x = x.transpose(1, 2)
        x = self.temporal_conv(x)
        x = x.transpose(1, 2)

        x = x[:, -1, :]
        x = self.grn(x)

        return self.output(x)


def load_data():
    data = np.load("data/train_test.npz")
    X_train_full = data["X_train"]
    y_train_full = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    n_samples = len(X_train_full)
    val_size = int(n_samples * VAL_RATIO)
    indices = np.random.permutation(n_samples)

    X_train = X_train_full[indices[val_size:]]
    y_train = y_train_full[indices[val_size:]]
    X_val = X_train_full[indices[:val_size]]
    y_val = y_train_full[indices[:val_size]]

    return X_train, y_train, X_val, y_val, X_test, y_test


def train_model():
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    n_features = X_train.shape[2]

    scaler_X = StandardScaler()
    X_train_flat = X_train.reshape(-1, n_features)
    X_train_scaled = scaler_X.fit_transform(X_train_flat).reshape(X_train.shape)

    X_val_flat = X_val.reshape(-1, n_features)
    X_val_scaled = scaler_X.transform(X_val_flat).reshape(X_val.shape)

    X_test_flat = X_test.reshape(-1, n_features)
    X_test_scaled = scaler_X.transform(X_test_flat).reshape(X_test.shape)

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_val_scaled = scaler_y.transform(y_val)
    y_test_scaled = scaler_y.transform(y_test)

    train_ds = TimeSeriesDataset(X_train_scaled, y_train_scaled)
    val_ds = TimeSeriesDataset(X_val_scaled, y_val_scaled)
    test_ds = TimeSeriesDataset(X_test_scaled, y_test_scaled)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimesNet(
        n_features=n_features,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        predict_len=PREDICT_STEPS,
        dropout=DROPOUT,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTimesNet Model: {total_params:,} parameters")
    print(f"Device: {device}")

    criterion = nn.HuberLoss(delta=0.5)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch)
                val_loss += criterion(pred, y_batch).item()

        val_loss /= len(val_loader)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "scaler_y_mean": scaler_y.mean_,
                    "scaler_y_scale": scaler_y.scale_,
                    "scaler_X_mean": scaler_X.mean_,
                    "scaler_X_scale": scaler_X.scale_,
                },
                "data/timesnet_model.pt",
            )
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:3d}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}"
            )

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    return model, scaler_y, device, X_test_scaled, y_test, scaler_X


def predict_and_visualize():
    print("\n" + "=" * 60)
    print("TIMESNET MODEL")
    print("=" * 60)

    model, y_scaler, device, X_test_scaled, y_test, scaler_X = train_model()

    checkpoint = torch.load("data/timesnet_model.pt", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        X_test_t = torch.FloatTensor(X_test_scaled).to(device)
        pred_scaled = model(X_test_t).cpu().numpy()

    pred = y_scaler.inverse_transform(pred_scaled)

    mse = np.mean((pred - y_test) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(pred - y_test))

    direction_actual = np.sign(y_test[:, 0])
    direction_pred = np.sign(pred[:, 0])
    direction_acc = (direction_actual == direction_pred).mean() * 100

    print(f"\n=== Test Results ===")
    print(f"MSE:                {mse:.8f}")
    print(f"RMSE:               {rmse:.8f}")
    print(f"MAE:                {mae:.8f}")
    print(f"Direction Accuracy: {direction_acc:.1f}%")

    df = pd.read_csv("data/eurusd_features.csv", parse_dates=["Date"], index_col="Date")
    train_end_idx = len(df) - len(y_test)

    current_close = df["Close"].values[train_end_idx : train_end_idx + len(pred)]
    actual_close = current_close + y_test[:, 0]
    predicted_close = current_close + pred[:, 0]

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    ax1 = axes[0]
    ax1.plot(actual_close, label="Actual Close", alpha=0.8, linewidth=1.5)
    ax1.plot(predicted_close, label="Predicted Close", alpha=0.8, linewidth=1.5)
    ax1.set_title("TimesNet: Actual vs Predicted Close Price (Test Data)")
    ax1.set_xlabel("Sample")
    ax1.set_ylabel("Close Price")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(current_close, label="Current Close (Input)", alpha=0.7, linewidth=1.5)
    ax2.plot(
        actual_close,
        label="Actual Next Close",
        alpha=0.7,
        linewidth=1.5,
        linestyle="--",
    )
    ax2.plot(
        predicted_close,
        label="Predicted Next Close",
        alpha=0.7,
        linewidth=1.5,
        linestyle=":",
    )
    ax2.set_title("TimesNet: Close Price Progression")
    ax2.set_xlabel("Sample")
    ax2.set_ylabel("Close Price")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("data/timesnet_prediction_plot.png", dpi=150, bbox_inches="tight")
    plt.close()

    np.save("data/timesnet_predictions.npy", pred)
    print("\nPlot saved to data/timesnet_prediction_plot.png")

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "direction_accuracy": direction_acc,
        "predicted_close": predicted_close,
        "actual_close": actual_close,
    }


if __name__ == "__main__":
    predict_and_visualize()
