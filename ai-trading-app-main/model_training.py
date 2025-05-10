from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
import xgboost as xgb
import pandas as pd

def train_models(df):
    df = df.copy()

    # Define a target: Predict if price increases > 0.5% in next 3 bars
    future_return = (df['Close'].shift(-3) - df['Close']) / df['Close']
    df['target'] = (future_return > 0.005).astype(int)
    df.dropna(inplace=True)

    X = df.drop(['Close', 'target'], axis=1)
    y = df['target']

    # ✅ SAFETY CHECK
    if len(X) < 50 or len(y) < 50:
        raise ValueError(f"Not enough data to train the model. Samples available: {len(X)}")

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
