import os
import json
import random
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, seq_len, d_model):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.01)
    def forward(self, x):
        return x + self.pos_embed[:, :x.size(1), :]

class FusionLSTMTransformer(nn.Module):
    def __init__(self, input_dim, lstm_hidden=64, trans_d=64,
                 trans_heads=4, trans_layers=2, seq_len=10, batch_first=True):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, lstm_hidden, num_layers=2, batch_first=batch_first)
        self.input_proj = nn.Linear(input_dim, trans_d)
        self.pos_emb = LearnablePositionalEmbedding(seq_len, trans_d)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=trans_d, nhead=trans_heads,
            dim_feedforward=trans_d*4, batch_first=batch_first
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=trans_layers)
        self.fc_fusion = nn.Sequential(
            nn.Linear(lstm_hidden + trans_d, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_last = lstm_out[:, -1, :]
        t = self.input_proj(x)
        t = self.pos_emb(t)
        t_out = self.transformer(t)
        t_last = t_out[:, -1, :]
        fused = torch.cat([lstm_last, t_last], dim=-1)
        out = self.fc_fusion(fused)
        return out

def create_dataset(data, window, target_idx):
    X, Y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        Y.append(data[i+window, target_idx])
    return np.array(X), np.array(Y)

def evaluate_metrics(y_true, y_pred):
    y_true = np.array(y_true).reshape(-1, 1)
    y_pred = np.array(y_pred).reshape(-1, 1)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mask = (np.abs(y_true) > 1e-8).astype(float)
    mape = (np.mean(np.abs((y_true - y_pred) / (np.where(mask, y_true, 1.0)))) * 100)
    r2 = r2_score(y_true, y_pred)
    return mse, mae, mape, r2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default=".")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu")

    df = pd.read_csv(args.data)
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True, errors='coerce')

    features = ['bid_price', 'bid_order_qty', 'bid_executed_qty', 'ask_order_qty', 'ask_executed_qty']
    df_features = df[['datetime'] + features].copy()
    df_features[features] = df_features[features].astype(float)

    scaler = MinMaxScaler()
    data_values = scaler.fit_transform(df_features[features].values)
    target_idx = features.index('ask_executed_qty')

    X, Y = create_dataset(data_values, args.window, target_idx)
    if len(X) == 0:
        print(json.dumps({"error": "not_enough_data"}))
        return

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    Y_tensor = torch.tensor(Y, dtype=torch.float32).view(-1, 1).to(device)

    input_dim = len(features)
    model = FusionLSTMTransformer(input_dim=input_dim, seq_len=args.window).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    model.train()
    for _ in range(args.epochs):
        pred = model(X_tensor)
        loss = criterion(pred, Y_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        y_pred = model(X_tensor).cpu().numpy()

    y_pred_inv = scaler.inverse_transform(np.hstack([y_pred, np.zeros((len(y_pred), input_dim - 1))]))[:, 0]
    y_true_inv = scaler.inverse_transform(np.hstack([Y_tensor.cpu().numpy(), np.zeros((len(Y_tensor), input_dim - 1))]))[:, 0]

    mse, mae, mape, r2 = evaluate_metrics(y_true_inv, y_pred_inv)
    last_pred = float(y_pred_inv[-1])
    std = float(np.std(y_pred_inv[-min(len(y_pred_inv), 100):])) if len(y_pred_inv) > 1 else 0.0
    upper = last_pred + 1.96 * std
    lower = last_pred - 1.96 * std

    os.makedirs(args.out_dir, exist_ok=True)
    result = {
        "predicted_value": last_pred,
        "confidence_score": max(0.0, min(1.0, 1.0 / (1.0 + mse))),
        "upper_bound": upper,
        "lower_bound": lower,
        "mse": mse,
        "mae": mae,
        "mape": mape,
        "r2": r2,
        "model_version": "fusion-lstm-trans-1.0.0"
    }
    print(json.dumps(result, ensure_ascii=False))

if __name__ == "__main__":
    main()
