import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb

# ===================== 1. 读取数据 =====================
raw_data = pd.read_csv('1/logistics_shipments_dataset.csv')
df = raw_data[raw_data["Status"] == "Delivered"].reset_index(drop=True)

# ===================== 2. 构建路线（分组关键字） =====================
df["route"] = (
    df["Origin_Warehouse"].astype(str) + "|" +
    df["Destination"].astype(str) + "|" +
    df["Carrier"].astype(str)
)

# ===================== 3. 对每条路线计算真实 min / max =====================
# 这是我们要让 XGBoost 去学习的目标
route_targets = df.groupby("route")["Transit_Days"].agg(
    route_min="min",
    route_max="max"
).reset_index()

# 把 route min/max 合并回原数据
df = df.merge(route_targets, on="route", how="left")

# ===================== 4. 特征 =====================
cat_features = ["Origin_Warehouse", "Destination", "Carrier"]
num_features = ["Weight_kg", "Cost", "Distance_miles"]
X = df[cat_features + num_features]

# 双目标：让 XGBoost 同时学 下限 和 上限
y_lower = df["route_min"]
y_upper = df["route_max"]

# ===================== 5. 数据集划分 =====================
X_train, X_test, yl_train, yl_test, yu_train, yu_test = train_test_split(
    X, y_lower, y_upper, test_size=0.3, random_state=42
)

# ===================== 6. 预处理流水线 =====================
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
    ("num", StandardScaler(), num_features)
])

# ===================== 7. XGBoost 训练：预测下限（min） =====================
model_min = Pipeline([
    ("pre", preprocessor),
    ("xgb", xgb.XGBRegressor(
        random_state=42,
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1
    ))
])
model_min.fit(X_train, yl_train)

# ===================== 8. XGBoost 训练：预测上限（max） =====================
# 关键：上限模型我们要偏向“保守”，让预测稍微大一点，降低 breach
model_max = Pipeline([
    ("pre", preprocessor),
    ("xgb", xgb.XGBRegressor(
        random_state=42,
        n_estimators=250,
        max_depth=6,
        learning_rate=0.1,
        objective='reg:squarederror'
    ))
])
model_max.fit(X_train, yu_train)

# ===================== 9. 模型预测 =====================
pred_min = model_min.predict(X_test)
pred_max = model_max.predict(X_test)

# 真实 Transit_Days
y_true = df.loc[X_test.index, "Transit_Days"].values

# ===================== 10. 业务指标评估（你最关心的） =====================
within = ((y_true >= pred_min) & (y_true <= pred_max)).mean()
breach = (y_true > pred_max).mean()
width = (pred_max - pred_min).mean()

print(f"✅ 区间覆盖率：{within:.2%}")
print(f"🚨 超过上限比例 (breach rate)：{breach:.2%}")
print(f"📏 平均区间宽度：{width:.2f} 天")