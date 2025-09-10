import os
import sys
import json
import argparse
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.manual_seed_all(seed)


class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, seq_len: int, d_model: int):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos_embed[:, :x.size(1), :]


class FusionLSTMTransformer(nn.Module):
    def __init__(self, input_dim: int, lstm_hidden: int = 64, trans_d: int = 64,
                 trans_heads: int = 4, trans_layers: int = 2, seq_len: int = 10):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, lstm_hidden, num_layers=2, batch_first=True)
        self.input_proj = nn.Linear(input_dim, trans_d)
        self.pos_emb = LearnablePositionalEmbedding(seq_len, trans_d)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=trans_d, nhead=trans_heads,
            dim_feedforward=trans_d * 4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=trans_layers)
        self.fc_fusion = nn.Sequential(
            nn.Linear(lstm_hidden + trans_d, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        lstm_last = lstm_out[:, -1, :]
        t = self.input_proj(x)
        t = self.pos_emb(t)
        t_out = self.transformer(t)
        t_last = t_out[:, -1, :]
        fused = torch.cat([lstm_last, t_last], dim=-1)
        return self.fc_fusion(fused)


def create_dataset(data: np.ndarray, window: int, target_index: int):
    X, Y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i + window])
        Y.append(data[i + window, target_index])
    return np.array(X), np.array(Y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out_dir", default="models/out")
    parser.add_argument("--window", type=int, default=10)
    args, _ = parser.parse_known_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(42)

    df = pd.read_csv(args.data)
    if "date_time" in df.columns:
        df["datetime"] = pd.to_datetime(df["date_time"], errors="coerce", utc=True)
    elif "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    else:
        df["datetime"] = pd.to_datetime("1970-01-01")
    df["datetime"] = df["datetime"].dt.tz_localize(None)

    features = ["bid_price", "bid_order_qty", "bid_executed_qty", "ask_order_qty", "ask_executed_qty"]
    for c in features:
        if c not in df.columns:
            df[c] = 0
    target_col = "ask_executed_qty"

    scaler = MinMaxScaler()
    data = scaler.fit_transform(df[features].values)
    X, Y = create_dataset(data, args.window, features.index(target_col))
    if len(X) == 0:
        print(json.dumps({"error": "not enough data after window"}))
        return

    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.float32).view(-1, 1)

    model = FusionLSTMTransformer(input_dim=len(features), seq_len=args.window)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    best = float("inf")
    wait = 0
    patience = 20
    for epoch in range(200):
        model.train()
        pred = model(X_tensor)
        loss = criterion(pred, Y_tensor)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if loss.item() < best:
            best = loss.item(); wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.eval()
    with torch.no_grad():
        y_pred = model(X_tensor).cpu().numpy()

    # inverse transform for the target dimension only
    y_pred_full = np.zeros((len(y_pred), len(features)))
    y_pred_full[:, features.index(target_col)] = y_pred.flatten()
    y_true_full = np.zeros((len(Y_tensor), len(features)))
    y_true_full[:, features.index(target_col)] = Y_tensor.cpu().numpy().flatten()
    y_pred_inv = scaler.inverse_transform(y_pred_full)[:, features.index(target_col)]
    y_true_inv = scaler.inverse_transform(y_true_full)[:, features.index(target_col)]

    mse = float(mean_squared_error(y_true_inv, y_pred_inv))
    mae = float(mean_absolute_error(y_true_inv, y_pred_inv))
    mape = float(np.mean(np.abs((y_true_inv - y_pred_inv) / (y_true_inv + 1e-8))) * 100)
    r2 = float(r2_score(y_true_inv, y_pred_inv))

    residuals = y_true_inv - y_pred_inv
    lower, upper = np.percentile(residuals, 1), np.percentile(residuals, 99)
    anoms = ((residuals < lower) | (residuals > upper)).astype(int)

    out_csv = os.path.join(args.out_dir, "anomaly_result.csv")
    df_out = df.iloc[args.window:].copy()
    df_out["is_anomaly"] = anoms
    df_out.to_csv(out_csv, index=False)

    # return a compact result for backend
    result = {
        "prediction_type": "regression_with_quantile_anomaly",
        "mse": mse, "mae": mae, "mape": mape, "r2": r2,
        "predicted_value": float(y_pred_inv[-1]) if len(y_pred_inv) > 0 else 0.0,
        "confidence_score": float(1.0),
    }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()


