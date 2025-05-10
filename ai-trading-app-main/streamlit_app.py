import streamlit as st
from data_loader import download_data
from feature_engineering import add_technical_indicators
from model_training import train_models
import matplotlib.pyplot as plt
import pandas as pd

st.title("AI Trading: Improved Model + Backtesting")

ticker = st.text_input("Enter Ticker:", "AAPL")

if st.button("Run Analysis"):
    with st.spinner("Downloading data..."):
        df = download_data(ticker)

    if df is None or df.empty or 'Close' not in df.columns:
        st.error("❌ Failed to download data or 'Close' column is missing.")
    else:
        st.write("✅ Raw Data", df.tail())

        with st.spinner("Engineering features..."):
            try:
                df = add_technical_indicators(df)

                # Add the target label directly in Streamlit
                future_return = (df['Close'].shift(-3) - df['Close']) / df['Close']
                df['target'] = (future_return > 0.005).astype(int)
                df.dropna(inplace=True)
                
                # SAFETY CHECK BEFORE TRAINING
                if df.empty or len(df) < 100:
                    st.error("❌ Not enough data to train the model. Try a different ticker or longer period.")
                    st.stop()
                
                st.write("🧠 Data with Features + Target", df.tail())
                st.write("🔍 Target Distribution", df['target'].value_counts(normalize=True))
            except Exception as e:
                st.error(f"❌ Feature engineering failed: {e}")
                st.stop()

        with st.spinner("Training model..."):
            try:
                model, scores = train_models(df)
                st.write("📊 Model Accuracy Scores", scores)

                features = df.drop(['Close', 'target'], axis=1)
                proba = model.predict_proba(features)[:, 1]
                df['signal'] = (proba > 0.6).astype(int)
                df['confidence'] = proba

                # Backtesting performance
                df['returns'] = df['Close'].pct_change()
                df['strategy'] = df['returns'] * df['signal'].shift(1)
                df[['returns', 'strategy']] = df[['returns', 'strategy']].fillna(0)

                df['cumulative_returns'] = (1 + df['returns']).cumprod()
                df['cumulative_strategy'] = (1 + df['strategy']).cumprod()

                st.line_chart(df[['cumulative_returns', 'cumulative_strategy']])
                st.success("✅ Model training and backtest complete!")

            except Exception as e:
                st.error(f"❌ Model training failed: {e}")
