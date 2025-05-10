import streamlit as st
from data_loader import download_data
from feature_engineering import add_technical_indicators
from model_training import train_models
import matplotlib.pyplot as plt
import pandas as pd

st.title("AI Trading: Interactive Threshold + Strategy Diagnostics")

ticker = st.text_input("Enter Ticker:", "BTC-USD")

threshold = st.slider("📈 Confidence threshold for signal", 0.3, 0.9, 0.6, 0.01)

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

        # Add target
        future_return = (df['Close'].shift(-3) - df['Close']) / df['Close']
        df['target'] = (future_return > 0.005).astype(int)

        # Cleanup
        df.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
        st.write("🧪 NaNs per column before dropna():")
        st.write(df.isna().sum())

        df.dropna(inplace=True)
        st.write(f"✅ Final usable rows before training: {len(df)}")

        if len(df) < 50:
            st.error("⚠️ Not enough data to train. Try a different ticker or longer time frame.")
            st.stop()

        with st.spinner("🤖 Training model..."):
            try:
                model, scores = train_models(df)
                st.write("📊 Model Accuracy Scores")
                st.write({k: round(v, 3) for k, v in scores.items()})

                features = df.drop(['Close', 'target'], axis=1)
                proba = model.predict_proba(features)[:, 1]
                df['signal'] = (proba > threshold).astype(int)
                df['confidence'] = proba

                df['returns'] = df['Close'].pct_change()
                df['strategy'] = df['returns'] * df['signal'].shift(1)
                df[['returns', 'strategy']] = df[['returns', 'strategy']].fillna(0)

                df['cumulative_returns'] = (1 + df['returns']).cumprod()
                df['cumulative_strategy'] = (1 + df['strategy']).cumprod()

                # ✅ True MultiIndex flattening
                df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

                st.write("📊 Final columns before plotting:")
                st.write(df.columns.tolist())

                # Diagnostics
                st.write("📊 Signal value counts:")
                st.write(df['signal'].value_counts())

                st.write("📊 Strategy return statistics:")
                st.write(df[['returns', 'strategy']].describe())

                if 'cumulative_returns' in df.columns and 'cumulative_strategy' in df.columns:
                    st.line_chart(df[['cumulative_returns', 'cumulative_strategy']])
                else:
                    st.warning("⚠️ Missing strategy columns.")

                st.success("✅ Model training and backtest complete!")
            except ValueError as ve:
                st.error(f"⚠️ {ve}")
            except Exception as e:
                st.error(f"❌ Model training failed: {e}")
