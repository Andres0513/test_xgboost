# ===================== 以下代码直接接在你后面运行 =====================
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 加载保存的数据集
df = pd.read_pickle("uninted_df.pkl")
print(f"✅ 数据集已加载：uninted_df.pkl")
# 删除临时字段
if '时间_dt' in df.columns:
    df = df.drop('时间_dt', axis=1)

# 取近期的数据去训练
start_date = '2024-06-01'
df = df[df['时间'].dt.date >= pd.to_datetime(start_date).date()]
# # 删掉没有燃气的行
# df = df[df['grid_env'] == 2]
# 按照日期选择最近的数据为验证集
split_test_1_date = '2026-04-16'
split_test_2_date = '2026-05-01'
test_2_df = df[df['时间'].dt.date >= pd.to_datetime(split_test_2_date).date()]
df = df[df['时间'].dt.date < pd.to_datetime(split_test_2_date).date()]
train_df, test_df = train_test_split(
    df,
    test_size=0.2,  # 0.2 作为测试集
    random_state=42  # 固定随机种子，保证每次划分结果一致
)
# 把所有特征强转成数字
for col in df.columns:
    train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
    test_df[col] = pd.to_numeric(test_df[col], errors='coerce')
    test_2_df[col] = pd.to_numeric(test_2_df[col], errors='coerce')
# 去掉出现NA的行
train_df = train_df.dropna().reset_index(drop=True)
test_df = test_df.dropna().reset_index(drop=True)

# X、y 划分
train_X = train_df.iloc[:, 4:]  # 第2列～最后：特征
train_y = train_df.iloc[:, 3]  # 第1列：要预测的目标
test_X = test_df.iloc[:, 4:]
test_y = test_df.iloc[:, 3]
test_2_X = test_2_df.iloc[:, 4:]
test_2_y = test_2_df.iloc[:, 3]

train_y = (train_y > 0).astype(int)
test_y = (test_y > 0).astype(int)
test_2_y = (test_2_y > 0).astype(int)

# print(train_y.head(10))
print(train_y.describe())

# 枚举特征：周几、季度、电网工况 → 转成 category 类型
categorical_cols = []
for col in train_X.columns:
    if '星期' in col or '季度' in col or 'grid_env' in col:
        categorical_cols.append(col)

# 转 category（XGBoost 支持直接训练）
train_X[categorical_cols] = train_X[categorical_cols].astype('category')
test_X[categorical_cols] = test_X[categorical_cols].astype('category')
test_2_X[categorical_cols] = test_2_X[categorical_cols].astype('category')

# 1. 把 category 列 转为 独热编码（MLP 必须用独热，不能用 category）
numerical_cols = [col for col in train_X.columns if col not in categorical_cols]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_cols),
    ("cat", OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
])

# 训练集拟合，测试集/验证集只转换（避免数据泄露）
X_train = preprocessor.fit_transform(train_X)
X_test = preprocessor.transform(test_X)
X_val = preprocessor.transform(test_2_X)

# 2. 转 PyTorch 张量
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)

y_train = torch.tensor(train_y.values, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(test_y.values, dtype=torch.float32).unsqueeze(1)
y_val = torch.tensor(test_2_y.values, dtype=torch.float32).unsqueeze(1)

# 3. MLP 模型（二分类）
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

input_dim = X_train.shape[1]
model = MLP(input_dim)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 4. 损失 & 优化器
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ===================== 训练：每一轮都看 训练/测试/验证 三个误差 =====================
epochs = 100
batch_size = 64

train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 移动到设备
X_test, y_test = X_test.to(device), y_test.to(device)
X_val, y_val = X_val.to(device), y_val.to(device)

print("\n开始训练，每轮输出三集损失与精度：\n")
for epoch in range(epochs):
    # === 训练 ===
    model.train()
    train_loss = 0
    for bx, by in train_loader:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        pred = model(bx)
        loss = criterion(pred, by)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # === 评估：训练集整体 ===
    model.eval()
    with torch.no_grad():
        train_pred = model(X_train.to(device))
        train_loss_all = criterion(train_pred, y_train.to(device)).item()
        train_acc = ((train_pred > 0.5) == y_train.to(device)).float().mean().item()

        # 测试集
        test_pred = model(X_test)
        test_loss = criterion(test_pred, y_test).item()
        test_acc = ((test_pred > 0.5) == y_test).float().mean().item()

        # 验证集（最关键！看泛化）
        val_pred = model(X_val)
        val_loss = criterion(val_pred, y_val).item()
        val_acc = ((val_pred > 0.5) == y_val).float().mean().item()

    # === 每轮输出 ===
    print(f"Epoch {epoch+1:2d} | "
          f"训练 loss: {train_loss_all:.4f} acc: {train_acc:.4f} | "
          f"测试 loss: {test_loss:.4f} acc: {test_acc:.4f} | "
          f"验证 loss: {val_loss:.4f} acc: {val_acc:.4f}")

# ===================== 训练结束后：在【验证集】上计算 正负一致性准确率 =====================
model.eval()
with torch.no_grad():
    # 预测概率
    val_pred_prob = model(X_val)
    # 转成 0/1 预测（>0.5 判为正，否则负）
    val_pred_class = (val_pred_prob > 0.5).float()

    # 真实标签
    val_true_class = y_val

    # 计算一致数量 & 一致率
    correct = (val_pred_class == val_true_class).sum().item()
    total = len(val_true_class)
    acc_percent = correct / total * 100

    # 展平成 numpy 方便看结果
    val_pred_np = val_pred_class.cpu().numpy().flatten()
    val_true_np = val_true_class.cpu().numpy().flatten()

# ===================== 输出最终验证集结果 =====================
print("\n" + "="*60)
print("🔎 【验证集】价差正负性预测结果")
print("="*60)
print(f"验证集总样本数：{total}")
print(f"预测正确数量：{correct}")
print(f"✅ 正负一致性准确率：{acc_percent:.2f} %")
print("="*60)

# 输出分类详细报告（可选）
from sklearn.metrics import classification_report
print("\n📊 验证集分类报告：")
print(classification_report(
    val_true_np, val_pred_np,
    target_names=["价差<=0（负）", "价差>0（正）"]
))