import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class TransformerTimeSeries(nn.Module):
    def __init__(self, input_dim=1, model_dim=64, num_heads=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(model_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.fc(x[-1])

def prepare_sequence_data(df, lookback=60):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['Close']].values)
    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i-lookback:i])
        y.append(scaled[i])
    X, y = np.array(X), np.array(y)
    return X, y, scaler

def train_transformer(X, y, model):
    X_tensor = torch.tensor(X, dtype=torch.float32).transpose(0, 1)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(5):
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output.squeeze(), y_tensor)
        loss.backward()
        optimizer.step()
    return model