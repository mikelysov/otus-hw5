import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

D_MODEL = 128
N_HEADS = 4
N_LAYERS = 2
DROPOUT = 0.1
BATCH_SIZE = 64
LEARNING_RATE = 0.001
MAX_EPOCHS = 100
PATIENCE = 15
VAL_RATIO = 0.15

data = np.load("data/train_test.npz")
X_train_full = data["X_train"]
y_train_full = data["y_train"]
X_test = data["X_test"]
y_test = data["y_test"]

n_samples = len(X_train_full)
val_size = int(n_samples * VAL_RATIO)
indices = np.random.permutation(n_samples)
val_indices = indices[:val_size]
train_indices = indices[val_size:]

X_train = X_train_full[train_indices]
y_train = y_train_full[train_indices]
X_val = X_train_full[val_indices]
y_val = y_train_full[val_indices]

n_features = X_train.shape[2]
n_timesteps = X_train.shape[1]
predict_len = y_train.shape[1]


def scale_data(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, n_features)
    X_train_scaled = scaler.fit_transform(X_train_flat).reshape(X_train.shape)

    X_val_flat = X_val.reshape(-1, n_features)
    X_val_scaled = scaler.transform(X_val_flat).reshape(X_val.shape)

    X_test_flat = X_test.reshape(-1, n_features)
    X_test_scaled = scaler.transform(X_test_flat).reshape(X_test.shape)

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


X_train_s, X_val_s, X_test_s, X_scaler = scale_data(X_train, X_val, X_test)

y_scaler = StandardScaler()
y_train_s = y_scaler.fit_transform(y_train)
y_val_s = y_scaler.transform(y_val)
y_test_s = y_scaler.transform(y_test)


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_ds = TimeSeriesDataset(X_train_s, y_train_s)
val_ds = TimeSeriesDataset(X_val_s, y_val_s)
test_ds = TimeSeriesDataset(X_test_s, y_test_s)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)


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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TemporalFusionTransformer(
    n_features,
    d_model=D_MODEL,
    n_heads=N_HEADS,
    n_layers=N_LAYERS,
    predict_len=predict_len,
    dropout=DROPOUT,
).to(device)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(
    f"\nModel: {sum(p.numel() for p in model.parameters())} total params, {total_params} trainable"
)
print(f"Device: {device}")

criterion = nn.HuberLoss(delta=0.5)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
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
        pred, _, _ = model(X_batch)
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
            pred, _, _ = model(X_batch)
            val_loss += criterion(pred, y_batch).item()

    val_loss /= len(val_loader)
    scheduler.step()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "y_scaler_mean": y_scaler.mean_,
                "y_scaler_scale": y_scaler.scale_,
            },
            "data/best_model.pt",
        )
        patience_counter = 0
    else:
        patience_counter += 1

    if epoch % 10 == 0:
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, lr={lr:.2e}"
        )

    if patience_counter >= PATIENCE:
        print(f"Early stopping at epoch {epoch}")
        break

checkpoint = torch.load("data/best_model.pt", weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
y_scaler.mean_ = checkpoint["y_scaler_mean"]
y_scaler.scale_ = checkpoint["y_scaler_scale"]
model.eval()

with torch.no_grad():
    X_test_t = torch.FloatTensor(X_test_s).to(device)
    pred_scaled, _, _ = model(X_test_t)
    pred_scaled = pred_scaled.cpu().numpy()

pred = y_scaler.inverse_transform(pred_scaled)

mse = np.mean((pred - y_test) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(pred - y_test))

print(f"\n=== Test Results ===")
print(f"MSE:  {mse:.8f}")
print(f"RMSE: {rmse:.8f}")
print(f"MAE:  {mae:.8f}")

direction_actual = np.sign(y_test[:, 0])
direction_pred = np.sign(pred[:, 0])
direction_acc = (direction_actual == direction_pred).mean() * 100
print(f"Direction Accuracy: {direction_acc:.1f}%")

np.save("data/predictions.npy", pred)
print(f"\nPredictions saved to data/predictions.npy")
