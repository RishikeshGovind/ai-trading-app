import matplotlib.pyplot as plt

def plot_patterns(df, title='Price with Patterns'):
    plt.figure(figsize=(14, 7))
    plt.plot(df['Close'], label='Close')
    if 'ema_20' in df.columns:
        plt.plot(df['ema_20'], label='EMA 20')
    if 'bb_bbh' in df.columns:
        plt.plot(df['bb_bbh'], label='BB High')
    if 'bb_bbl' in df.columns:
        plt.plot(df['bb_bbl'], label='BB Low')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()