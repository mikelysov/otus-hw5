#!/usr/bin/env python3
"""
Скрипт для создания обучающей и тестовой выборок из данных с техническими индикаторами.
Формирует последовательности для обучения нейросетевых моделей временных рядов.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ============ Константы ============
RANDOM_SEED = 42  # Сид для воспроизводимости результатов
LOOKBACK = 200  # Количество исторических тиков на вход модели
PREDICT_STEPS = 1  # Количество шагов предсказания (1 = следующий тик)
TRAIN_END_DATE = "2025-01-01"  # Дата разделения train/test

# Колонки, которые не используются как признаки
EXCLUDE_COLS = ["Date", "Close", "High", "Low", "Open", "Volume"]

# Устанавливаем сид для воспроизводимости
np.random.seed(RANDOM_SEED)

# Загружаем данные с признаками
print("Loading features...")
df = pd.read_csv("data/eurusd_features.csv", parse_dates=["Date"], index_col="Date")
print(f"Loaded {len(df)} rows")

# Разделяем данные на train и test по дате
train_df = df[df.index < TRAIN_END_DATE].copy()
test_df = df[df.index >= TRAIN_END_DATE].copy()

print(f"Train: {len(train_df)} ({train_df.index.min()} - {train_df.index.max()})")
print(f"Test: {len(test_df)} ({test_df.index.min()} - {test_df.index.max()})")

# Получаем список признаков (все колонки кроме исключённых)
feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
print(f"Features ({len(feature_cols)}): {feature_cols[:5]}...")


def create_sequences(data, close_col, lookback, predict_steps):
    """
    Создаёт последовательности для обучения/тестирования.

    Параметры:
        data: DataFrame с данными
        close_col: имя колонки с ценой закрытия
        lookback: количество исторических тиков
        predict_steps: количество шагов предсказания

    Возвращает:
        X: массив входных последовательностей (n_samples, lookback, n_features)
        y: массив целевых значений - дельт цен (n_samples, predict_steps)
    """
    X, y = [], []
    target_idx = data.columns.get_loc(close_col)
    close_values = data[close_col].values

    # Проходим по всем возможным позициям
    for i in range(lookback, len(data) - predict_steps + 1):
        # Входные данные: последние lookback тиков
        X.append(data.iloc[i - lookback : i].values)

        # Целевое значение: изменение цены close
        # Дельта = следующая цена - текущая цена
        deltas = (
            close_values[i : i + predict_steps]
            - close_values[i - 1 : i + predict_steps - 1]
        )
        y.append(deltas)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# Создаём последовательности для train и test
X_train, y_train = create_sequences(train_df, "Close", LOOKBACK, PREDICT_STEPS)
X_test, y_test = create_sequences(test_df, "Close", LOOKBACK, PREDICT_STEPS)

print(f"\nX_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

# Сохраняем данные в формате npz
np.savez(
    "data/train_test.npz",
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    feature_cols=np.array(feature_cols),
)
print(f"\nSaved to data/train_test.npz")
