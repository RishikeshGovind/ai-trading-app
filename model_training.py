from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
import xgboost as xgb
import pandas as pd

def train_models(df):
    df = df.copy()

    # Add target again for completeness (should already exist)
    future_return = (df['Close'].shift(-3) - df['Close']) / df['Close']
    df['target'] = (future_return > 0.005).astype(int)

    df.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
    df.dropna(inplace=True)

    X = df.drop(['Close', 'target'], axis=1)
    y = df['target']

    # DEBUG
    print("X shape before align:", X.shape)
    print("y shape before align:", y.shape)
    print("NaNs in X:", X.isna().sum().sum())
    print("NaNs in y:", y.isna().sum())

    # Align features and labels to avoid length mismatch
    X, y = X.align(y, join='inner', axis=0)

    if len(X) < 50:
        raise ValueError(f"Not enough samples to train. Final shape: X={X.shape}, y={y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'rf': RandomForestClassifier(),
        'gb': GradientBoostingClassifier(),
        'lr': LogisticRegression(max_iter=500),
        'xgb': xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False)
    }

    scores = {}
    for name, model in models.items():
        try:
            scores[name] = cross_val_score(model, X_train, y_train, cv=5).mean()
        except Exception as e:
            scores[name] = 0
            print(f"Model {name} failed: {e}")

    top_models = sorted(scores, key=scores.get, reverse=True)[:3]
    ensemble = VotingClassifier(estimators=[(name, models[name]) for name in top_models], voting='soft')
    ensemble.fit(X_train, y_train)

    return ensemble, scores
