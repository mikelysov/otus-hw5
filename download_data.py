import yfinance as yf
import pandas as pd
from datetime import datetime

ticker = "EURUSD=X"
start_date = "2020-01-01"
end_date = datetime.now().strftime("%Y-%m-%d")

data = yf.download(ticker, start=start_date, end=end_date, interval="1d")

data.columns = [col[0] for col in data.columns]
data.to_csv("data/eurusd.csv")
print(f"Скачано {len(data)} записей с {start_date} по {end_date}")
print(f"Период: {data.index.min()} - {data.index.max()}")
print(f"\nПервые 5 строк:")
print(data.head())
print(f"\nКолонки: {list(data.columns)}")
