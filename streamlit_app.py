import streamlit as st
from data_loader import download_data
from feature_engineering import add_technical_indicators
from model_training import train_models  
import matplotlib.pyplot as plt
import pandas as pd

st.title("AI Trading: Full Debug Streamlit App")

ticker = st.text_input("Enter Ticker:", "BTC-USD")

if st.button("Run Analysis"):
    with st.spinner("📥 Downloading data..."):
        df = download_data(ticker)

    if df is None or df.empty or 'Close' not in df.columns:
        st.error("❌ Failed to download data or 'Close' column is missing.")
    else:
        st.write(f"✅ Rows after download: {len(df)}")
        st.dataframe(df.tail())

        with st.spinner("⚙️ Engineering features..."):
            try:
                df = add_technical_indicators(df)
                st.write(f"✅ Rows after feature engineering: {len(df)}")
                st.dataframe(df.tail())
            except Exception as e:
                st.error(f"❌ Feature engineering failed: {e}")
                st.stop()

        # Add target after indicators
        future_return = (df['Close'].shift(-3) - df['Close']) / df['Close']
        df['target'] = (future_return > 0.005).astype(int)

        # Replace infinities and drop missing values
        df.replace([float('inf'), float('-inf')], pd.NA, inplace=True)

        st.write("🧪 NaNs per column before dropna():")
        st.write(df.isna().sum())

        df.dropna(inplace=True)

        st.write(f"✅ Final usable rows before training: {len(df)}")
        st.dataframe(df.head())

        with st.spinner("🤖 Training model..."):
            try:
                model, scores = train_models(df)
                st.write("📊 Model Accuracy Scores")
                st.write(scores)

                features = df.drop(['Close', 'target'], axis=1)
                proba = model.predict_proba(features)[:, 1]
                df['signal'] = (proba > 0.6).astype(int)
                df['confidence'] = proba

                df['returns'] = df['Close'].pct_change()
                df['strategy'] = df['returns'] * df['signal'].shift(1)
                df[['returns', 'strategy']] = df[['returns', 'strategy']].fillna(0)

                df['cumulative_returns'] = (1 + df['returns']).cumprod()
                df['cumulative_strategy'] = (1 + df['strategy']).cumprod()

                st.line_chart(df[['cumulative_returns', 'cumulative_strategy']])
                st.success("✅ Model training and backtest complete!")
            except ValueError as ve:
                st.error(f"⚠️ {ve}")
            except Exception as e:
                st.error(f"❌ Model training failed: {e}")
