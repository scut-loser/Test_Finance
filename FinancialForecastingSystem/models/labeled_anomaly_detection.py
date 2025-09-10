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
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_curve, auc,
    accuracy_score, recall_score, f1_score
)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def create_dataset(X: np.ndarray, y: np.ndarray, window: int):
    X_seq, y_seq = [], []
    for i in range(len(X) - window):
        X_seq.append(X[i:i + window])
        y_seq.append(y[i + window])
    return np.array(X_seq), np.array(y_seq)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="input csv path")
    parser.add_argument("--out_dir", default="models/out", help="output directory")
    parser.add_argument("--window", type=int, default=10)
    args, unknown = parser.parse_known_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(42)

    df = pd.read_csv(args.data)
    # column compatibility
    if "date_time" in df.columns:
        df["datetime"] = pd.to_datetime(df["date_time"], errors="coerce")
    elif "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    else:
        # fabricate if not exists
        df["datetime"] = pd.to_datetime("1970-01-01")

    features = ["bid_price", "bid_order_qty", "bid_executed_qty", "ask_order_qty", "ask_executed_qty"]
    for c in features:
        if c not in df.columns:
            df[c] = 0
    target_col = "is_anomaly"
    if target_col not in df.columns:
        # fallback: no labels → mark all as 0
        df[target_col] = 0

    X = df[features].values.astype(float)
    y = df[target_col].values.astype(int)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_seq, y_seq = create_dataset(X_scaled, y, args.window)
    if len(X_seq) == 0:
        print(json.dumps({"error": "not enough data after window"}))
        return

    X_tensor = torch.tensor(X_seq, dtype=torch.float32)
    y_tensor = torch.tensor(y_seq, dtype=torch.float32).view(-1, 1)

    model = FusionLSTMTransformer(input_dim=len(features), seq_len=args.window)
    num_total = len(y_seq)
    num_anomaly = int(y_seq.sum()) if int(y_seq.sum()) > 0 else 1
    num_normal = num_total - num_anomaly
    pos_weight = torch.tensor([num_normal / num_anomaly])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best = float("inf")
    patience = 20
    wait = 0
    for epoch in range(200):
        model.train()
        output = model(X_tensor)
        loss = criterion(output, y_tensor)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if loss.item() + 1e-8 < best:
            best = loss.item(); wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.eval()
    with torch.no_grad():
        logits = model(X_tensor).squeeze().cpu().numpy()
        probs = torch.sigmoid(torch.tensor(logits)).numpy()

    precision, recall, thresholds = precision_recall_curve(y_seq, probs)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = int(np.argmax(f1_scores))
    best_threshold = float(thresholds[best_idx]) if thresholds.size > 0 else 0.5
    y_pred = (probs >= best_threshold).astype(int)

    acc = float(accuracy_score(y_seq, y_pred))
    rec = float(recall_score(y_seq, y_pred))
    f1v = float(f1_score(y_seq, y_pred))

    # Save artifacts
    out_pred = os.path.join(args.out_dir, "anomaly_points_pred.csv")
    df_pred = df.iloc[args.window:].copy()
    df_pred["pred_prob"] = probs
    df_pred["pred_label"] = y_pred
    df_pred.to_csv(out_pred, index=False)

    result = {
        "prediction_type": "anomaly_detection",
        "best_threshold": best_threshold,
        "accuracy": acc,
        "recall": rec,
        "f1": f1v,
        # map to generic fields consumed by backend
        "predicted_value": float(probs[-1]) if len(probs) > 0 else 0.0,
        "confidence_score": float(1.0 - abs(probs[-1] - best_threshold)) if len(probs) > 0 else 0.0
    }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()


