"""
Скрипт для загрузки исторических данных EUR/USD через yfinance.
Данные сохраняются в CSV формат для дальнейшей обработки.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime

# Параметры загрузки данных
ticker = "EURUSD=X"  # Тикер пары EUR/USD на yfinance
start_date = "2020-01-01"  # Начальная дата
end_date = datetime.now().strftime("%Y-%m-%d")  # Конечная дата (сегодня)

# Загрузка данных через yfinance
data = yf.download(ticker, start=start_date, end=end_date, interval="1d")

# Убираем мультииндекс в колонках (оставляем только имена колонок)
data.columns = [col[0] for col in data.columns]

# Сохранение в CSV
data.to_csv("data/eurusd.csv")

# Вывод информации о загруженных данных
print(f"Скачано {len(data)} записей с {start_date} по {end_date}")
print(f"Период: {data.index.min()} - {data.index.max()}")
print(f"\nПервые 5 строк:")
print(data.head())
print(f"\nКолонки: {list(data.columns)}")
