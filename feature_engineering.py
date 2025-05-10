import ta
import pandas as pd

def add_technical_indicators(df):
    close = df['Close'].squeeze()

    df['rsi'] = ta.momentum.RSIIndicator(close).rsi()
    df['ema_20'] = ta.trend.EMAIndicator(close, window=20).ema_indicator()
    df['ema_slope'] = df['ema_20'].diff()

    # Commented out heavy indicators to reduce NaNs
    # df['macd'] = ta.trend.MACD(close).macd()
    # bb = ta.volatility.BollingerBands(close)
    # df['bb_bbm'] = bb.bollinger_mavg()
    # df['bb_bbh'] = bb.bollinger_hband()
    # df['bb_bbl'] = bb.bollinger_lband()

    df['volatility'] = close.rolling(window=10).std()
    df['returns'] = close.pct_change()
    df['hour'] = df.index.hour
    df['volume_change'] = df['Volume'].pct_change()

    df.dropna(inplace=True)
    return df