from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd

class MLBasedStrategy(Strategy):
    def init(self):
        self.data = self.data.df.copy()
        self.model = self.model_instance

    def next(self):
        if len(self.data) < 20:
            return

        row = self.data.iloc[-1].drop(['Open', 'High', 'Low', 'Close', 'Volume'])
        signal = self.model.predict([row])[0]

        if signal == 1 and not self.position:
            self.buy()
        elif signal == 0 and self.position:
            self.sell()

def run_backtest(df, model):
    from backtesting import Backtest
    df = df.copy().dropna()
    bt = Backtest(df, MLBasedStrategy, cash=10000, commission=.002)
    bt.model_instance = model
    stats = bt.run()
    return stats, bt