import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import warnings

warnings.filterwarnings('ignore')

# ===================== M芯片设置 =====================
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"✅ 使用设备: {device} (M芯片适配)")

# ===================== 加载数据 =====================
df = pd.read_pickle("uninted_df.pkl")
df = df.sort_values("时间").reset_index(drop=True)

if '时间_dt' in df.columns:
    df = df.drop('时间_dt', axis=1)

# ===================== 时间切分 =====================
split_val = "2026-05-01"
valid_mask = df["时间"] >= pd.to_datetime(split_val)
train_mask = df["时间"] < pd.to_datetime(split_val)

df_train = df[train_mask].copy()
df_val = df[valid_mask].copy()

# ===================== 特征配置 =====================
FEAT_COLS = list(df.columns[4:])
LABEL_COL = df.columns[3]
SEQ_LEN = 60  # 看过去90天

# 分类列 / 数值列
categorical_cols = [c for c in FEAT_COLS if "星期" in c or "季度" in c or "grid_env" in c]
numerical_cols = [c for c in FEAT_COLS if c not in categorical_cols]

# 获取列索引（修复关键！！！）
feature_df = df[FEAT_COLS]
cat_indices = [feature_df.columns.get_loc(c) for c in categorical_cols]
num_indices = [feature_df.columns.get_loc(c) for c in numerical_cols]

# 预处理（用索引，不用列名，修复报错）
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_indices),
    ("cat", OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_indices)
])

# ===================== 构造序列 =====================
def build_sequences(df):
    X, y = [], []
    df = df.reset_index(drop=True)
    for i in range(SEQ_LEN, len(df)):
        seq = df[FEAT_COLS].iloc[i-SEQ_LEN:i].values
        label = 1 if df[LABEL_COL].iloc[i] > 0 else 0
        X.append(seq)
        y.append(label)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

X_train, y_train = build_sequences(df_train)
X_val, y_val = build_sequences(df_val)

# ===================== 归一化 =====================
B, S, D = X_train.shape
X_train_reshaped = X_train.reshape(-1, D)
X_val_reshaped = X_val.reshape(-1, D)

X_train_processed = preprocessor.fit_transform(X_train_reshaped).reshape(B, SEQ_LEN, -1)
X_val_processed = preprocessor.transform(X_val_reshaped).reshape(len(X_val), SEQ_LEN, -1)

# ===================== 转为张量 =====================
X_train = torch.tensor(X_train_processed, dtype=torch.float32)
X_val = torch.tensor(X_val_processed, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

# ===================== 注意力模型 =====================
class AttentionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, 1, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + attn_out)

class History90DModel(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.proj = nn.Linear(feature_dim, 64)
        self.attn1 = AttentionBlock(64)
        self.attn2 = AttentionBlock(64)
        self.cls = nn.Sequential(
            nn.Linear(64, 32), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(32, 1), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.proj(x)
        x = self.attn1(x)
        x = self.attn2(x)
        x = x.mean(dim=1)
        return self.cls(x)

# ===================== 训练 =====================
model = History90DModel(feature_dim=X_train.shape[-1]).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-4)
criterion = nn.BCELoss()

best_acc = 0
patience = 10
counter = 0

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_train, y_train),
    batch_size=32, shuffle=True
)

print("\n🚀 90天历史相似日注意力模型 开始训练\n")

for epoch in range(60):
    model.train()
    for bx, by in train_loader:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        loss = criterion(model(bx), by)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        train_pred = model(X_train.to(device))
        train_acc = ((train_pred > 0.5) == y_train.to(device)).float().mean().item()

        val_pred = model(X_val.to(device))
        val_acc = ((val_pred > 0.5) == y_val.to(device)).float().mean().item()

    if val_acc > best_acc:
        best_acc = val_acc
        counter = 0
        torch.save(model.state_dict(), "best_90d_model.pth")
    else:
        counter += 1
        if counter >= patience:
            print(f"\n⏹️ 早停 | 最佳验证精度: {best_acc:.4f}")
            break

    print(f"Epoch {epoch+1:2d} | 训练:{train_acc:.4f} | 验证:{val_acc:.4f}")

# ===================== 最终结果 =====================
model.load_state_dict(torch.load("best_90d_model.pth"))
model.eval()
with torch.no_grad():
    train_final = ((model(X_train.to(device)) > 0.5) == y_train.to(device)).float().mean().item() * 100
    val_final = ((model(X_val.to(device)) > 0.5) == y_val.to(device)).float().mean().item() * 100

print("\n" + "="*60)
print("🎯 90天历史相似日注意力模型 最终成绩")
print(f"训练集: {train_final:.2f}%")
print(f"验证集: {val_final:.2f}%")
print("="*60)