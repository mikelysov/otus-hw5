#!/usr/bin/env python3
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42
LOOKBACK = 200
PREDICT_STEPS = 1
TRAIN_END_DATE = "2025-01-01"

EXCLUDE_COLS = ["Date", "Close", "High", "Low", "Open", "Volume"]

np.random.seed(RANDOM_SEED)

print("Loading features...")
df = pd.read_csv("data/eurusd_features.csv", parse_dates=["Date"], index_col="Date")
print(f"Loaded {len(df)} rows")

train_df = df[df.index < TRAIN_END_DATE].copy()
test_df = df[df.index >= TRAIN_END_DATE].copy()

print(f"Train: {len(train_df)} ({train_df.index.min()} - {train_df.index.max()})")
print(f"Test: {len(test_df)} ({test_df.index.min()} - {test_df.index.max()})")

feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
print(f"Features ({len(feature_cols)}): {feature_cols[:5]}...")


def create_sequences(data, close_col, lookback, predict_steps):
    X, y = [], []
    target_idx = data.columns.get_loc(close_col)
    close_values = data[close_col].values

    for i in range(lookback, len(data) - predict_steps + 1):
        X.append(data.iloc[i - lookback : i].values)
        deltas = (
            close_values[i : i + predict_steps]
            - close_values[i - 1 : i + predict_steps - 1]
        )
        y.append(deltas)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


X_train, y_train = create_sequences(train_df, "Close", LOOKBACK, PREDICT_STEPS)
X_test, y_test = create_sequences(test_df, "Close", LOOKBACK, PREDICT_STEPS)

print(f"\nX_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

np.savez(
    "data/train_test.npz",
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    feature_cols=np.array(feature_cols),
)
print(f"\nSaved to data/train_test.npz")
