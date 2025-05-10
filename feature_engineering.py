import ta
import pandas as pd

def add_technical_indicators(df):
    close = df['Close'].squeeze()

    # Add basic indicators
    df['rsi'] = ta.momentum.RSIIndicator(close).rsi()
    df['ema_20'] = ta.trend.EMAIndicator(close, window=20).ema_indicator()
    df['ema_slope'] = df['ema_20'].diff()

    # heavier indicators to be uncommented later
    # df['macd'] = ta.trend.MACD(close).macd()
    # bb = ta.volatility.BollingerBands(close)
    # df['bb_bbm'] = bb.bollinger_mavg()
    # df['bb_bbh'] = bb.bollinger_hband()
    # df['bb_bbl'] = bb.bollinger_lband()

    # Additional features
    df['volatility'] = close.rolling(window=10).std()
    df['returns'] = df['Close'].pct_change().clip(-1, 1)
    df['hour'] = df.index.hour
    df['volume_change'] = df['Volume'].pct_change().clip(-10, 10)

    # ✅ Clean infinities and NaNs
    df.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
    df.dropna(inplace=True)

    return df
