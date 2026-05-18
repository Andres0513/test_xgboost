import pandas as pd

# 先确保两个 df 的时间列都是 datetime 类型
df_1['时间'] = pd.to_datetime(df_1['时间'])
df_3['时间_dt'] = pd.to_datetime(df_3['时间_dt'])

# 给 df_1 按时间排序，方便后面快速查找
df_1 = df_1.sort_values('时间').reset_index(drop=True)

# 初始化新列，先填 NaN
for i in range(8):
    df_3[f'日前竞价空间{i}'] = pd.NA
    df_3[f'统调负荷预测{i}'] = pd.NA

# 遍历 df_3 的每一行
for idx, target_time in df_3['时间_dt'].items():
    # 筛选条件：时间 >= (target_time - 2h) 且 时间 < target_time
    mask = (df_1['时间'] >= target_time - pd.Timedelta(hours=2)) & (df_1['时间'] < target_time)
    window_data = df_1.loc[mask, ['日前竞价空间', '统调负荷预测']].copy()

    # 按时间从近到远排序（倒序），这样第一个元素就是离 target_time 最近的
    window_data = window_data.sort_values(df_1.columns[df_1.columns.get_loc('时间')], ascending=False)

    # 取最多8条，不足8条后面会保留 NA
    for i in range(min(8, len(window_data))):
        df_3.at[idx, f'日前竞价空间{i}'] = window_data['日前竞价空间'].iloc[i]
        df_3.at[idx, f'统调负荷预测{i}'] = window_data['统调负荷预测'].iloc[i]



# ===================== 以下代码直接接在你后面运行 =====================
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 加载保存的数据集
df = pd.read_pickle("uninted_df.pkl")
print(f"✅ 数据集已加载：uninted_df.pkl")
# 删除临时字段
if '时间_dt' in df.columns:
    df = df.drop('时间_dt', axis=1)

# 取近期的数据去训练
start_date = '2024-06-01'
df = df[df['时间'].dt.date >= pd.to_datetime(start_date).date()]
# 按照日期选择最近的数据为验证集
split_validation_data = '2026-05-01'
validation_df = df[df['时间'].dt.date >= pd.to_datetime(split_validation_data).date()]
df = df[df['时间'].dt.date < pd.to_datetime(split_validation_data).date()]
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)
# 把所有特征强转成数字
for col in df.columns:
    train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
    test_df[col] = pd.to_numeric(test_df[col], errors='coerce')
    validation_df[col] = pd.to_numeric(validation_df[col], errors='coerce')
# 去掉出现NA的行
train_df = train_df.dropna().reset_index(drop=True)
test_df = test_df.dropna().reset_index(drop=True)

# X、y 划分
train_X = train_df.iloc[:, 4:]
train_y = train_df.iloc[:, 3]
test_X = test_df.iloc[:, 4:]
test_y = test_df.iloc[:, 3]
validation_X = validation_df.iloc[:, 4:]
validation_y = validation_df.iloc[:, 3]

train_y = (train_y > 0).astype(int)
test_y = (test_y > 0).astype(int)
validation_y = (validation_y > 0).astype(int)

# 枚举特征
categorical_cols = []
for col in train_X.columns:
    if '星期' in col or '季度' in col or 'grid_env' in col:
        categorical_cols.append(col)

train_X[categorical_cols] = train_X[categorical_cols].astype('category')
test_X[categorical_cols] = test_X[categorical_cols].astype('category')
validation_X[categorical_cols] = validation_X[categorical_cols].astype('category')

numerical_cols = [col for col in train_X.columns if col not in categorical_cols]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_cols),
    ("cat", OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
])

X_train = preprocessor.fit_transform(train_X)
X_test = preprocessor.transform(test_X)
X_val = preprocessor.transform(validation_X)

# 转张量
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)

y_train = torch.tensor(train_y.values, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(test_y.values, dtype=torch.float32).unsqueeze(1)
y_val = torch.tensor(validation_y.values, dtype=torch.float32).unsqueeze(1)

# ===================== 🔥 优化后模型：防过拟合 =====================
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.4),  # 防止过拟合

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

input_dim = X_train.shape[1]
model = MLP(input_dim)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.BCELoss()
# ✅ weight_decay = 正则化，防止过拟合
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)

# ===================== 🔥 早停策略 =====================
epochs = 100
batch_size = 64
best_val_acc = 0  # 记录最好的验证精度
patience = 50     # 10轮不提升就停
counter = 0

train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

X_test, y_test = X_test.to(device), y_test.to(device)
X_val, y_val = X_val.to(device), y_val.to(device)

print("\n开始训练（早停 + 正则化 防过拟合）\n")

for epoch in range(epochs):
    # 训练
    model.train()
    for bx, by in train_loader:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        pred = model(bx)
        loss = criterion(pred, by)
        loss.backward()
        optimizer.step()

    # 评估
    model.eval()
    with torch.no_grad():
        train_pred = model(X_train.to(device))
        train_acc = ((train_pred > 0.5) == y_train.to(device)).float().mean().item()

        test_pred = model(X_test)
        test_acc = ((test_pred > 0.5) == y_test).float().mean().item()

        val_pred = model(X_val)
        val_acc = ((val_pred > 0.5) == y_val).float().mean().item()

    # ===================== 🔥 早停逻辑 =====================
    if test_acc > best_val_acc:
        best_val_acc = test_acc
        counter = 0
        torch.save(model.state_dict(), "best_model.pth")  # 保存最优模型
    else:
        counter += 1

    if counter >= patience:
        print(f"\n⏹️ 早停！最佳验证精度: {best_val_acc:.4f}")
        break

    print(f"Epoch {epoch+1:2d} | 训练:{train_acc:.4f} | 测试:{test_acc:.4f} | 验证:{val_acc:.4f}")

# 加载最优模型
model.load_state_dict(torch.load("best_model.pth"))

# ===================== 最终评估 =====================
model.eval()
with torch.no_grad():
    # 训练集
    yp_train = model(X_train.to(device))
    train_correct = ((yp_train > 0.5) == y_train.to(device)).float().sum().item()
    train_total = len(y_train)
    train_acc = train_correct / train_total * 100

    # 测试集
    yp_test = model(X_test)
    test_correct = ((yp_test > 0.5) == y_test).float().sum().item()
    test_total = len(y_test)
    test_acc = test_correct / test_total * 100

    # 验证集
    yp_val = model(X_val)
    val_correct = ((yp_val > 0.5) == y_val).float().sum().item()
    val_total = len(y_val)
    val_acc = val_correct / val_total * 100

    val_pred_np = (yp_val > 0.5).cpu().numpy().flatten()
    val_true_np = y_val.cpu().numpy().flatten()

print("\n" + "="*60)
print("🎯 最终 三数据集 正负预测一致率")
print("="*60)
print(f"训练集 | {train_correct}/{train_total} | {train_acc:.2f}%")
print(f"测试集 | {test_correct}/{test_total} | {test_acc:.2f}%")
print(f"验证集 | {val_correct}/{val_total} | {val_acc:.2f}%")
print("="*60)

print("\n📊 验证集报告")
print(classification_report(val_true_np, val_pred_np, target_names=["负", "正"]))