import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data/train_test.npz"
MODEL_PATH = "data/best_model.pt"


class GatedResidualNetwork(nn.Module):
    def __init__(self, d_model, d_ff=None, dropout=0.1, epsilon=1e-5):
        super().__init__()
        d_ff = d_ff or d_model * 4
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.linear3 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model, eps=epsilon)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = self.linear3(x)
        x = self.dropout(torch.relu(self.linear1(x)))
        x = self.linear2(x)
        gate = torch.sigmoid(self.linear3(x))
        return self.norm(gate * residual + (1 - gate) * x)


class VariableSelectionNetwork(nn.Module):
    def __init__(self, n_features, d_model, dropout=0.1):
        super().__init__()
        self.feature_weights = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_features),
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        weights = self.softmax(self.feature_weights(x.mean(dim=1)))
        return x, weights


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(1)].unsqueeze(0)


class TemporalFusionTransformer(nn.Module):
    def __init__(
        self,
        n_features,
        d_model=128,
        n_heads=4,
        n_layers=2,
        predict_len=10,
        dropout=0.1,
    ):
        super().__init__()
        self.n_features = n_features
        self.predict_len = predict_len
        self.d_model = d_model

        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        self.vsn = VariableSelectionNetwork(n_features, d_model, dropout)

        self.encoder_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model,
                    n_heads,
                    dim_feedforward=d_model * 4,
                    dropout=dropout,
                    batch_first=True,
                    activation="gelu",
                )
                for _ in range(n_layers)
            ]
        )

        self.temporal_attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.attention_norm = nn.LayerNorm(d_model)

        self.decoder = nn.Sequential(
            GatedResidualNetwork(d_model, dropout=dropout),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, predict_len),
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)

        x, feature_weights = self.vsn(x)

        for layer in self.encoder_layers:
            x = layer(x)

        attn_out, attn_weights = self.temporal_attention(x, x, x)
        x = self.attention_norm(x + attn_out)

        x = x[:, -1, :]
        x = self.decoder(x)

        return x, attn_weights, feature_weights


def load_model_and_scaler(model_path, n_features, predict_len):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalFusionTransformer(
        n_features=n_features,
        d_model=128,
        n_heads=4,
        n_layers=2,
        predict_len=predict_len,
        dropout=0.1,
    ).to(device)

    checkpoint = torch.load(model_path, weights_only=False, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    y_scaler = StandardScaler()
    y_scaler.mean_ = checkpoint["y_scaler_mean"]
    y_scaler.scale_ = checkpoint["y_scaler_scale"]

    return model, y_scaler, device


def predict_next_close(df_features, n_samples=None):
    data = np.load(DATA_PATH)
    X = data["X_test"]
    y_test = data["y_test"]

    n_features = X.shape[2]
    predict_len = y_test.shape[1]

    if n_samples is None:
        n_samples = len(X)

    model, y_scaler, device = load_model_and_scaler(MODEL_PATH, n_features, predict_len)

    scaler = StandardScaler()
    X_flat = X.reshape(-1, n_features)
    scaler.fit(X_flat)
    X_scaled = scaler.transform(X_flat).reshape(X.shape)

    X_batch = torch.FloatTensor(X_scaled[:n_samples]).to(device)

    with torch.no_grad():
        pred_scaled = model(X_batch)[0].cpu().numpy()

    pred_delta = y_scaler.inverse_transform(pred_scaled)

    train_end_idx = len(df_features) - len(y_test)

    results = []
    for i in range(n_samples):
        current_close = df_features.iloc[train_end_idx + i]["Close"]

        if predict_len == 1:
            delta = pred_delta[i, 0]
        else:
            delta = np.sum(pred_delta[i])

        predicted_close = current_close + delta
        actual_delta = y_test[i, 0]
        actual_close = current_close + actual_delta

        results.append(
            {
                "sample": i,
                "current_close": current_close,
                "predicted_delta": delta,
                "actual_delta": actual_delta,
                "predicted_close": predicted_close,
                "actual_close": actual_close,
                "direction_correct": np.sign(delta) == np.sign(actual_delta),
            }
        )

    return pd.DataFrame(results)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    df = pd.read_csv("data/eurusd_features.csv", parse_dates=["Date"], index_col="Date")

    print("=" * 60)
    print("MODEL VERIFICATION - Predicted vs Actual Close Prices")
    print("=" * 60)

    results = predict_next_close(df)

    print(results.head(20).to_string(index=False))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Direction Accuracy: {results['direction_correct'].mean() * 100:.1f}%")
    print(f"Mean Predicted Close: {results['predicted_close'].mean():.6f}")
    print(f"Mean Actual Close:    {results['actual_close'].mean():.6f}")
    print(
        f"Mean Close Error:     {(results['predicted_close'] - results['actual_close']).abs().mean():.6f}"
    )

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    ax1 = axes[0]
    ax1.plot(
        results["actual_close"].values, label="Actual Close", alpha=0.8, linewidth=1.5
    )
    ax1.plot(
        results["predicted_close"].values,
        label="Predicted Close",
        alpha=0.8,
        linewidth=1.5,
    )
    ax1.set_title("Actual vs Predicted Close Price (Test Data)")
    ax1.set_xlabel("Sample")
    ax1.set_ylabel("Close Price")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(
        results["current_close"].values,
        label="Current Close (Input)",
        alpha=0.7,
        linewidth=1.5,
    )
    ax2.plot(
        results["actual_close"].values,
        label="Actual Next Close",
        alpha=0.7,
        linewidth=1.5,
        linestyle="--",
    )
    ax2.plot(
        results["predicted_close"].values,
        label="Predicted Next Close",
        alpha=0.7,
        linewidth=1.5,
        linestyle=":",
    )
    ax2.set_title("Close Price Progression: Current -> Actual/Predicted")
    ax2.set_xlabel("Sample")
    ax2.set_ylabel("Close Price")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("data/close_prediction_plot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\nPlot saved to data/close_prediction_plot.png")
