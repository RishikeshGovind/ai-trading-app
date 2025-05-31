import streamlit as st
from data_loader import download_data
from feature_engineering import add_technical_indicators
from model_training import train_models
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(
    page_title="AI Trading App",
    layout="wide"
)

st.title("📈 AI-Powered Stock Strategy Evaluator")

st.markdown("""
This app helps you evaluate a simple AI-based trading strategy compared to the market.

It uses machine learning to decide **when to buy or stay out** of a stock or crypto.  
We'll show you:
- The historical price trend
- The model's trading signals
- How well the strategy performs vs simply holding
- Key performance metrics anyone can understand
""")

ticker = st.text_input("Enter a stock or crypto symbol (e.g., BTC-USD, AAPL):", "BTC-USD")
threshold = st.slider("🎯 Prediction confidence threshold (higher = fewer trades)", 0.3, 0.9, 0.6, 0.01)

if st.button("🔍 Analyze"):
    with st.spinner("⏳ Downloading price data..."):
        df = download_data(ticker)

    if df is None or df.empty:
        st.error("❌ Could not download data.")
    else:
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

        if 'Close' not in df.columns:
            st.error("❌ 'Close' column not found.")
            st.write("Columns:", df.columns.tolist())
            st.stop()

        st.subheader("📉 Recent Price History")
        st.line_chart(df['Close'])

        with st.spinner("⚙️ Engineering technical features..."):
            try:
                df = add_technical_indicators(df)
            except Exception as e:
                st.error(f"❌ Feature engineering failed: {e}")
                st.stop()

        # Add target
        future_return = (df['Close'].shift(-3) - df['Close']) / df['Close']
        df['target'] = (future_return > 0.005).astype(int)

        df.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
        df.dropna(inplace=True)

        if len(df) < 50:
            st.error("⚠️ Not enough data to evaluate.")
            st.stop()

        with st.spinner("🧠 Training AI model..."):
            try:
                model, scores = train_models(df)

                features = df.drop(['Close', 'target'], axis=1)
                proba = model.predict_proba(features)[:, 1]
                df['signal'] = (proba > threshold).astype(int)
                df['confidence'] = proba

                df['returns'] = df['Close'].pct_change()
                df['strategy'] = df['returns'] * df['signal'].shift(1)
                df[['returns', 'strategy']] = df[['returns', 'strategy']].fillna(0)
                df['cumulative_returns'] = (1 + df['returns']).cumprod()
                df['cumulative_strategy'] = (1 + df['strategy']).cumprod()

                st.subheader("📊 Strategy vs Market Performance")
                st.line_chart(df[['cumulative_returns', 'cumulative_strategy']])

                # Compute metrics
                strategy = df['strategy'].dropna()
                market = df['returns'].dropna()
                def safe_sharpe(r): return (r.mean() / r.std()) * (252**0.5) if r.std() > 0 else 0

                metrics = pd.DataFrame({
                    'Metric': ['Total Return', 'Volatility', 'Sharpe Ratio', 'Win Rate'],
                    'Strategy': [
                        f"{(1 + strategy).prod() - 1:.2%}",
                        f"{strategy.std() * (252 ** 0.5):.2%}",
                        f"{safe_sharpe(strategy):.2f}",
                        f"{(strategy > 0).mean():.2%}"
                    ],
                    'Market': [
                        f"{(1 + market).prod() - 1:.2%}",
                        f"{market.std() * (252 ** 0.5):.2%}",
                        f"{safe_sharpe(market):.2f}",
                        f"{(market > 0).mean():.2%}"
                    ]
                })

                st.subheader("📋 Key Performance Metrics")
                st.dataframe(metrics)

                st.markdown("""
#### 📘 How to Read This:

- **Total Return**: how much you'd gain/loss from strategy vs just holding
- **Volatility**: how bumpy the returns are (lower = smoother)
- **Sharpe Ratio**: return vs risk (higher = better)
- **Win Rate**: how often the strategy made a profit on trades

Adjust the slider above to control how confident the model must be before triggering a trade.
""")
                st.success("✅ Done! Explore the results above.")
            except Exception as e:
                st.error(f"❌ Model training or evaluation failed: {e}")
