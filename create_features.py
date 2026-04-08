#!/usr/bin/env python3
"""
Скрипт для расчёта технических индикаторов на основе исторических данных EUR/USD.
Создаёт признаки для обучения нейросетевых моделей.
"""

import pandas as pd
import numpy as np
import ta
from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator, CCIIndicator
from ta.momentum import (
    RSIIndicator,
    StochasticOscillator,
    WilliamsRIndicator,
    ROCIndicator,
)
from ta.volume import (
    ChaikinMoneyFlowIndicator,
    OnBalanceVolumeIndicator,
    ForceIndexIndicator,
)

# Загрузка данных из CSV
print("Loading data...")
df = pd.read_csv("data/eurusd.csv", parse_dates=["Date"], index_col="Date")
print(f"Loaded {len(df)} rows")

# Извлекаем ценовые данные
close = df["Close"]
high = df["High"]
low = df["Low"]
volume = df["Volume"]

# Создаём копию DataFrame для добавления признаков
result = df.copy()
print("Computing features...")

# ============ Ценовые дельты (изменения цены) ============
# Разница между текущей и предыдущей ценой закрытия
result["diff_close"] = close.diff()
result["diff_high"] = high.diff()
result["diff_low"] = low.diff()
result["diff_open"] = df["Open"].diff()
result["diff_volume"] = volume.diff()

# ============ Процентные изменения ============
# Относительное изменение цены в процентах
result["pct_change_close"] = close.pct_change()
result["pct_change_high"] = high.pct_change()
result["pct_change_low"] = low.pct_change()
result["pct_change_open"] = df["Open"].pct_change()
# Для volume используем fillna(0), т.к. Volume=0 для forex данных
result["pct_change_volume"] = volume.pct_change().fillna(0)

# ============ Логарифмическая доходность ============
# Логарифм отношения цен (используется в финансовой математике)
result["log_return"] = np.log(close / close.shift(1))

# ============ Простые скользящие средние (SMA) ============
# Среднее значение цены за последние N периодов
result["sma_5"] = SMAIndicator(close, window=5).sma_indicator()
result["sma_10"] = SMAIndicator(close, window=10).sma_indicator()
result["sma_20"] = SMAIndicator(close, window=20).sma_indicator()
result["sma_50"] = SMAIndicator(close, window=50).sma_indicator()

# ============ Экспоненциальные скользящие средние (EMA) ============
# Взвешенное среднее с большим весом на недавние цены
result["ema_5"] = EMAIndicator(close, window=5).ema_indicator()
result["ema_10"] = EMAIndicator(close, window=10).ema_indicator()
result["ema_20"] = EMAIndicator(close, window=20).ema_indicator()
result["ema_50"] = EMAIndicator(close, window=50).ema_indicator()

# ============ RSI (Relative Strength Index) ============
# Осциллятор momentum, показывающий силу тренда (0-100)
result["rsi"] = RSIIndicator(close, window=14).rsi()

# ============ Bollinger Bands (Полосы Боллинджера) ============
# Волатильность рынка на основе стандартного отклонения
bb = BollingerBands(close, window=20, window_dev=2)
result["bb_upper"] = bb.bollinger_hband()  # Верхняя граница
result["bb_mid"] = bb.bollinger_mavg()  # Средняя линия (SMA)
result["bb_lower"] = bb.bollinger_lband()  # Нижняя граница
result["bb_width"] = (result["bb_upper"] - result["bb_lower"]) / result[
    "bb_mid"
]  # Ширина полос
result["bb_position"] = (close - result["bb_lower"]) / (  # Позиция цены в полосах
    result["bb_upper"] - result["bb_lower"]
)

# ============ MACD (Moving Average Convergence Divergence) ============
# Трендовый осциллятор на основе двух EMA
macd = MACD(close)
result["macd"] = macd.macd()  # Линия MACD
result["macd_signal"] = macd.macd_signal()  # Сигнальная линия
result["macd_diff"] = macd.macd_diff()  # Гистограмма MACD

# ============ ADX (Average Directional Index) ============
# Индекс среднего направления движения (сила тренда)
adx = ADXIndicator(high, low, close, window=14)
result["adx"] = adx.adx()  # Общий ADX
result["adx_neg"] = adx.adx_neg()  # Отрицательное направленное движение
result["adx_pos"] = adx.adx_pos()  # Положительное направленное движение

# ============ CMF (Chaikin Money Flow) ============
# Индикатор давления покупателей/продавцов
cmf = ChaikinMoneyFlowIndicator(
    high, low, close, volume, window=20
).chaikin_money_flow()
result["cmf"] = cmf.fillna(0)

# ============ Momentum и ROC (Rate of Change) ============
# Скорость изменения цены
result["momentum"] = ROCIndicator(close, window=10).roc()  # Momentum
result["roc"] = ROCIndicator(close, window=12).roc()  # Rate of Change

# ============ ATR (Average True Range) ============
# Средний истинный диапазон (мера волатильности)
result["atr"] = AverageTrueRange(high, low, close, window=14).average_true_range()

# ============ Stochastic Oscillator (Стохастический осциллятор) ============
# Сравнивает цену закрытия с диапазоном максимум-минимум
stoch = StochasticOscillator(high, low, close, window=14, smooth_window=3)
result["stoch_k"] = stoch.stoch()  # %K линия
result["stoch_d"] = stoch.stoch_signal()  # %D сигнальная линия

# ============ Williams %R ============
# Осциллятор перекупленности/перепроданности
result["williams_r"] = WilliamsRIndicator(high, low, close, lbp=14).williams_r()

# ============ OBV (On Balance Volume) ============
# Кумулятивный индикатор объёма
result["obv"] = OnBalanceVolumeIndicator(close, volume).on_balance_volume()

# ============ Force Index ============
# Сила движения на основе цены и объёма
result["force_index"] = ForceIndexIndicator(close, volume, window=13).force_index()

# ============ CCI (Commodity Channel Index) ============
# Индекс товарного канала (трендовый осциллятор)
result["cci"] = CCIIndicator(high, low, close, window=20).cci()

# ============ SMA 200 и производные ============
# Долгосрочная скользящая средняя
sma_200 = SMAIndicator(close, window=200).sma_indicator()
result["sma_200"] = sma_200
result["price_sma200_ratio"] = close / sma_200  # Отношение цены к SMA200

# ============ EMA 12/26 и производные ============
# Краткосрочные и среднесрочные EMA
ema_12 = EMAIndicator(close, window=12).ema_indicator()
ema_26 = EMAIndicator(close, window=26).ema_indicator()
result["ema_12"] = ema_12
result["ema_26"] = ema_26
result["ema_12_26_ratio"] = ema_12 / ema_26  # Отношение EMA12 к EMA26

# ============ Дополнительные признаки ============
# Отношения между ценами
result["high_low_ratio"] = high / low  # Отношение максимума к минимуму
result["close_open_ratio"] = close / df["Open"]  # Отношение close к open

# Удаляем строки с NaN (из-за начальных значений индикаторов)
print(f"Before dropna: {len(result)}")
result = result.dropna()
print(f"After dropna: {len(result)}")

# Сохраняем результат в CSV
result.to_csv("data/eurusd_features.csv")

# Выводим статистику
print(f"\nДобавлено признаков: {len(result.columns) - len(df.columns)}")
print(f"Всего записей: {len(result)}")
print(f"Период: {result.index.min()} - {result.index.max()}")
print(f"\nКолонки:\n{list(result.columns)}")
