#!/usr/bin/env python3
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

print("Loading data...")
df = pd.read_csv("data/eurusd.csv", parse_dates=["Date"], index_col="Date")
print(f"Loaded {len(df)} rows")

close = df["Close"]
high = df["High"]
low = df["Low"]
volume = df["Volume"]

close = df["Close"]
high = df["High"]
low = df["Low"]
volume = df["Volume"]

result = df.copy()
print("Computing features...")

result["diff_close"] = close.diff()
result["diff_high"] = high.diff()
result["diff_low"] = low.diff()
result["diff_open"] = df["Open"].diff()
result["diff_volume"] = volume.diff()

result["pct_change_close"] = close.pct_change()
result["pct_change_high"] = high.pct_change()
result["pct_change_low"] = low.pct_change()
result["pct_change_open"] = df["Open"].pct_change()
result["pct_change_volume"] = volume.pct_change().fillna(0)

result["log_return"] = np.log(close / close.shift(1))

result["sma_5"] = SMAIndicator(close, window=5).sma_indicator()
result["sma_10"] = SMAIndicator(close, window=10).sma_indicator()
result["sma_20"] = SMAIndicator(close, window=20).sma_indicator()
result["sma_50"] = SMAIndicator(close, window=50).sma_indicator()

result["ema_5"] = EMAIndicator(close, window=5).ema_indicator()
result["ema_10"] = EMAIndicator(close, window=10).ema_indicator()
result["ema_20"] = EMAIndicator(close, window=20).ema_indicator()
result["ema_50"] = EMAIndicator(close, window=50).ema_indicator()

result["rsi"] = RSIIndicator(close, window=14).rsi()

bb = BollingerBands(close, window=20, window_dev=2)
result["bb_upper"] = bb.bollinger_hband()
result["bb_mid"] = bb.bollinger_mavg()
result["bb_lower"] = bb.bollinger_lband()
result["bb_width"] = (result["bb_upper"] - result["bb_lower"]) / result["bb_mid"]
result["bb_position"] = (close - result["bb_lower"]) / (
    result["bb_upper"] - result["bb_lower"]
)

macd = MACD(close)
result["macd"] = macd.macd()
result["macd_signal"] = macd.macd_signal()
result["macd_diff"] = macd.macd_diff()

adx = ADXIndicator(high, low, close, window=14)
result["adx"] = adx.adx()
result["adx_neg"] = adx.adx_neg()
result["adx_pos"] = adx.adx_pos()

cmf = ChaikinMoneyFlowIndicator(
    high, low, close, volume, window=20
).chaikin_money_flow()
result["cmf"] = cmf.fillna(0)

result["momentum"] = ROCIndicator(close, window=10).roc()
result["roc"] = ROCIndicator(close, window=12).roc()

result["atr"] = AverageTrueRange(high, low, close, window=14).average_true_range()

stoch = StochasticOscillator(high, low, close, window=14, smooth_window=3)
result["stoch_k"] = stoch.stoch()
result["stoch_d"] = stoch.stoch_signal()

result["williams_r"] = WilliamsRIndicator(high, low, close, lbp=14).williams_r()

result["obv"] = OnBalanceVolumeIndicator(close, volume).on_balance_volume()

result["force_index"] = ForceIndexIndicator(close, volume, window=13).force_index()

result["cci"] = CCIIndicator(high, low, close, window=20).cci()

sma_200 = SMAIndicator(close, window=200).sma_indicator()
result["sma_200"] = sma_200
result["price_sma200_ratio"] = close / sma_200

ema_12 = EMAIndicator(close, window=12).ema_indicator()
ema_26 = EMAIndicator(close, window=26).ema_indicator()
result["ema_12"] = ema_12
result["ema_26"] = ema_26
result["ema_12_26_ratio"] = ema_12 / ema_26

result["high_low_ratio"] = high / low
result["close_open_ratio"] = close / df["Open"]

print(f"Before dropna: {len(result)}")
result = result.dropna()
print(f"After dropna: {len(result)}")

result.to_csv("data/eurusd_features.csv")

print(f"\nДобавлено признаков: {len(result.columns) - len(df.columns)}")
print(f"Всего записей: {len(result)}")
print(f"Период: {result.index.min()} - {result.index.max()}")
print(f"\nКолонки:\n{list(result.columns)}")
