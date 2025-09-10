import os
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

# -------------------- 配置参数 --------------------
SEED = 42
DAYS_FOR_TRAIN = 10
EPOCHS = 200
LR = 1e-3
LSTM_HIDDEN = 64
TRANS_D_MODEL = 64
TRANS_HEADS = 4
TRANS_LAYERS = 2
BATCH_FIRST = True
PATIENCE = 20  # 早停耐心值
OUT_DIR = "."
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------- 设置随机种子 --------------------
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed()

# -------------------- 加载数据 --------------------
df = pd.read_csv(r"anomaly_result.csv")
df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

features = ["bid_price", "bid_order_qty", "bid_executed_qty", "ask_order_qty", "ask_executed_qty"]
target_col = "is_anomaly"

X = df[features].values.astype(float)
y = df[target_col].values.astype(int)

# -------------------- 归一化 --------------------
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# -------------------- 构造序列数据 --------------------
def create_dataset(X, y, window):
    X_seq, y_seq = [], []
    for i in range(len(X) - window):
        X_seq.append(X[i:i+window])
        y_seq.append(y[i+window])
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_dataset(X_scaled, y, DAYS_FOR_TRAIN)
X_tensor = torch.tensor(X_seq, dtype=torch.float32)
y_tensor = torch.tensor(y_seq, dtype=torch.float32).view(-1, 1)

# -------------------- 模型结构 --------------------
class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, seq_len, d_model):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.01)
    def forward(self, x):
        return x + self.pos_embed[:, :x.size(1), :]

class FusionLSTMTransformer(nn.Module):
    def __init__(self, input_dim, lstm_hidden=LSTM_HIDDEN, trans_d=TRANS_D_MODEL,
                 trans_heads=TRANS_HEADS, trans_layers=TRANS_LAYERS, seq_len=DAYS_FOR_TRAIN):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, lstm_hidden, num_layers=2, batch_first=BATCH_FIRST)
        self.input_proj = nn.Linear(input_dim, trans_d)
        self.pos_emb = LearnablePositionalEmbedding(seq_len, trans_d)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=trans_d, nhead=trans_heads,
            dim_feedforward=trans_d * 4, batch_first=BATCH_FIRST
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
        return self.fc_fusion(fused)

# -------------------- 初始化模型 --------------------
input_dim = len(features)
model = FusionLSTMTransformer(input_dim=input_dim)

# -------------------- 类别加权 BCE --------------------
num_total = len(y_seq)
num_anomaly = sum(y_seq)
num_normal = num_total - num_anomaly
pos_weight = torch.tensor([num_normal / num_anomaly])
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# -------------------- 训练（带早停） --------------------
losses = []
best_loss = float('inf')
trigger_times = 0

for epoch in range(EPOCHS):
    model.train()
    output = model(X_tensor)
    loss = criterion(output, y_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    # 早停逻辑
    if loss.item() < best_loss - 1e-6:
        best_loss = loss.item()
        trigger_times = 0
    else:
        trigger_times += 1

    if trigger_times >= PATIENCE:
        print(f"\n⚠️ 早停触发！Epoch {epoch+1} 停止训练。")
        break

    if (epoch + 1) % 10 == 0:
        print(f"[FusionLSTMTransformer] Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.6f}")

# -------------------- 保存 Loss 曲线 --------------------
plt.figure()
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("FusionLSTMTransformer Training Loss")
plt.savefig(os.path.join(OUT_DIR, "loss_fusion_lstm_transformer.png"))
plt.close()

# -------------------- 预测 --------------------
model.eval()
with torch.no_grad():
    logits = model(X_tensor).squeeze().cpu().numpy()
    probs = torch.sigmoid(torch.tensor(logits)).numpy()

# -------------------- Precision-Recall 曲线 & 阈值 --------------------
precision, recall, thresholds = precision_recall_curve(y_seq, probs)
f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]

# -------------------- 分类评估 --------------------
y_pred = (probs >= best_threshold).astype(int)
accuracy = accuracy_score(y_seq, y_pred)
recall_score_val = recall_score(y_seq, y_pred)
f1_val = f1_score(y_seq, y_pred)

print(f"\n📊 最佳阈值 threshold : {best_threshold:.4f}")
print(f" - Accuracy : {accuracy:.4f}")
print(f" - Recall   : {recall_score_val:.4f}")
print(f" - F1 Score : {f1_val:.4f}")
print(classification_report(y_seq, y_pred, target_names=["Normal", "Anomaly"]))
print("混淆矩阵：\n", confusion_matrix(y_seq, y_pred))

# -------------------- 保存预测结果 --------------------
df_pred = df.iloc[DAYS_FOR_TRAIN:].copy()
df_pred["pred_prob"] = probs
df_pred["pred_label"] = y_pred
df_pred.to_csv("anomaly_pred_fusion_lstm_transformer.csv", index=False)

df_anomalies = df_pred[df_pred["pred_label"] == 1][["datetime", "pred_prob", "pred_label"]]
df_anomalies.to_csv("anomaly_points_pred.csv", index=False)

# -------------------- 可视化 --------------------
plt.figure(figsize=(12,6))
plt.plot(df_pred["datetime"], df_pred["pred_prob"], label="Predicted Anomaly Probability")
plt.scatter(df_pred["datetime"][df_pred["pred_label"]==1],
            df_pred["pred_prob"][df_pred["pred_label"]==1],
            color='red', label="Detected Anomalies")
plt.axhline(best_threshold, color='orange', linestyle='--', label=f'Threshold={best_threshold:.4f}')
plt.xlabel("Datetime")
plt.ylabel("Anomaly Probability")
plt.title("Anomaly Detection Probability & Detected Anomalies")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("anomaly_detection_visualization.png")
plt.close()

print("\n✅ 修正版异常检测完成！生成文件：")
print(" - loss_fusion_lstm_transformer.png")
print(" - anomaly_pred_fusion_lstm_transformer.csv")
print(" - anomaly_points_pred.csv")
print(" - anomaly_detection_visualization.png")
