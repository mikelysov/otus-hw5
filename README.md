# EUR/USD Price Prediction with Neural Networks

Прогнозирование изменения цены закрытия EUR/USD с использованием нейросетевых архитектур TFT и TimesNet.

## Описание

Модель предсказывает изменение цены close на следующем тике на основе 200 исторических значений с техническими индикаторами.

## Структура проекта

```
├── download_data.py      # Загрузка данных EUR/USD
├── create_features.py    # Расчёт технических индикаторов
├── create_dataset.py     # Создание train/test наборов
├── train_tft.py         # Обучение TFT модели
├── train_timesnet.py    # Обучение TimesNet модели
├── predict_close.py      # Функция предсказания close цены
├── analyze.py           # Анализ и метрики
└── data/
    ├── eurusd.csv              # Исходные данные OHLCV
    ├── eurusd_features.csv     # Данные с индикаторами
    ├── train_test.npz          # Обучающая выборка
    ├── best_model.pt           # TFT модель
    ├── timesnet_model.pt       # TimesNet модель
    └── *.png                   # Графики анализа
```

## Установка

```bash
pip install -r requirements.txt
```

Или используйте виртуальное окружение:
```bash
source .venv/bin/activate
```

## Быстрый старт

### 1. Загрузка данных
```bash
python download_data.py
```

### 2. Расчёт признаков
```bash
python create_features.py
```

### 3. Создание набора данных
```bash
python create_dataset.py
```

### 4. Обучение моделей

**TFT (Temporal Fusion Transformer):**
```bash
python train_tft.py
```

**TimesNet:**
```bash
python train_timesnet.py
```

### 5. Анализ результатов
```bash
python analyze.py
```

### 6. Предсказание close цены
```bash
python predict_close.py
```

## Архитектуры моделей

### TFT (Temporal Fusion Transformer)
- **Параметры:** 658,877
- **Компоненты:**
  - Variable Selection Network (VSN)
  - Gated Residual Network (GRN)
  - Multi-head Temporal Attention
  - Positional Encoding

### TimesNet
- **Параметры:** 63,969
- **Компоненты:**
  - Inception Block с несколькими ветвями сверток
  - Temporal Convolution
  - Gated Residual Network

## Результаты

| Модель | MSE | RMSE | MAE | Direction Accuracy | R² |
|--------|-----|------|-----|-------------------|-----|
| **TFT** | 0.00001275 | 0.00357 | 0.00258 | **76.0%** | 0.34 |
| TimesNet | 0.00001385 | 0.00372 | 0.00277 | 73.6% | - |

## Признаки (51)

### Ценовые дельты и проценты
- diff_close, diff_high, diff_low, diff_open, diff_volume
- pct_change_close, pct_change_high, pct_change_low, pct_change_open, pct_change_volume
- log_return

### Moving Averages
- SMA: 5, 10, 20, 50, 200
- EMA: 5, 10, 20, 50, 12, 26

### Осцилляторы
- RSI (14)
- Stochastic (K, D)
- Williams %R
- CCI (20)
- Momentum, ROC

### Волатильность
- Bollinger Bands (upper, mid, lower, width, position)
- ATR (14)

### Тренд
- MACD (signal, diff)
- ADX, ADX+, ADX-

### Объём
- CMF, OBV, Force Index
- high_low_ratio, close_open_ratio
- price_sma200_ratio, ema_12_26_ratio

## Периоды данных

- **Train:** 2020-10-06 - 2024-12-31 (906 сэмплов)
- **Test:** 2025-01-02 - 2026-04-07 (125 сэмплов)
- **Lookback:** 200 тиков
- **Prediction:** 1 шаг (изменение цены close)

## Зависимости

- numpy
- pandas
- torch
- scikit-learn
- matplotlib
- ta (technical analysis)
- yfinance
