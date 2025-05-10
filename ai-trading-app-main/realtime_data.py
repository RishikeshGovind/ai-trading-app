import yfinance as yf
import pandas as pd
import time

def stream_data(ticker, interval='1m', duration_minutes=30):
    end_time = time.time() + duration_minutes * 60
    all_data = []

    print(f"Streaming {ticker} data for {duration_minutes} minutes...")
    while time.time() < end_time:
        df = yf.download(tickers=ticker, interval=interval, period='1d')
        latest = df.iloc[-1:]
        all_data.append(latest)
        print(latest)
        time.sleep(60)

    result = pd.concat(all_data)
    result.drop_duplicates(inplace=True)
    return result