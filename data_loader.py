import yfinance as yf
import pandas as pd

def download_data(ticker, interval="1h", period="120d"):
    df = yf.download(ticker, interval=interval, period=period)
    if df.empty:
        return None
    df.dropna(inplace=True)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    return df