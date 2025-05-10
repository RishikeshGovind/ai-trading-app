# 🤖 AI-Powered Trading Strategy Explorer

This project is a complete end-to-end AI-based trading strategy evaluator built using Python and Streamlit.  
It allows users to analyse stock or crypto tickers and compare an ML-driven trading strategy vs the actual market performance.

## 📦 Features

- ✅ Live data download (e.g., BTC-USD, AAPL, ETH-USD)
- 📈 Technical indicators added as features
- 🧠 Machine learning model ensemble (Random Forest, Gradient Boosting, XGBoost)
- 🎯 Signal prediction with adjustable confidence threshold
- 💹 Backtesting: strategy vs market performance
- 📊 Performance metrics table (Sharpe Ratio, Volatility, Win Rate, Return)
- 🧠 Layman-friendly UI with clear explanations

---

## 🚀 How to Use

1. **Clone the repo** or upload files to [Streamlit Cloud](https://streamlit.io/cloud)
2. Run the app:
   ```bash
   streamlit run streamlit_app.py
   ```
3. Type in a ticker (e.g. `BTC-USD`) and press "Analyze"
4. Adjust the confidence threshold slider to control signal sensitivity

---

## 📘 Strategy Logic

- The app trains an ensemble ML model to predict whether the price will increase > 0.5% in the next 3 bars.
- A trading **signal = 1** is generated if the model is confident above the selected threshold.
- Returns are calculated by applying these signals with one-bar delay (`signal.shift(1)`).
- Strategy performance is compared with simply holding the asset.

---

## 📊 Sample Metrics Table

| Metric         | Strategy | Market |
|----------------|----------|--------|
| Total Return   |  +80%    | -21%   |
| Volatility     |  Low     |  High  |
| Sharpe Ratio   |  1.42    |  0.91  |
| Win Rate       |  65%     |  55%   |

---

## 📂 Folder Structure

```
📁 ai-trading-app/
├── streamlit_app.py         # Main app logic
├── data_loader.py           # Data download from Yahoo Finance
├── feature_engineering.py   # Technical indicator creation
├── model_training.py        # ML model training + ensemble
├── requirements.txt         # Dependencies
```

---

## 🔧 Requirements

Install packages:
```bash
pip install -r requirements.txt
```

---

## 📜 License

MIT License---

## 🤖 How the Strategy Works

This strategy uses **machine learning** to decide when to buy an asset — based purely on patterns in past data.

### 🔍 Step-by-Step:

1. **Feature Engineering**
   - We create technical indicators like RSI, EMA, momentum, volatility, etc.
   - These give the model clues about market behavior.

2. **Target Definition**
   - The model learns to predict: _“Will the price rise > 0.5% in the next 3 bars?”_
   - We label past data accordingly: `1` for a rise, `0` otherwise.

3. **Training ML Models**
   - We use a mix of models: Random Forest, Gradient Boosting, and XGBoost.
   - The top 3 performers are combined into a VotingClassifier ensemble (like a team of advisors).

4. **Generating Buy Signals**
   - When the model is confident (e.g., 60%+), it issues a **buy signal**.
   - You can adjust the confidence threshold to be more/less aggressive.

5. **Backtesting**
   - We simulate how this strategy would have performed historically.
   - It’s compared with simply holding the asset, showing both on a cumulative return graph.

### 📈 Why Use This?
- Avoids buying during uncertain times
- Can reduce drawdowns (loss periods)
- Adapts to new patterns over time via retraining

The result: a **data-driven, rule-based strategy** that reacts to real market signals — not gut feelings.

---
