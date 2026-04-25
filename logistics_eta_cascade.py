import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, accuracy_score, roc_auc_score
import xgboost as xgb

# ===================== 1. 读取数据 =====================
raw_data = pd.read_csv('1/logistics_shipments_dataset.csv')

# 只保留已送达的数据
df = raw_data[raw_data["Status"] == "Delivered"].reset_index(drop=True)

# ----------------------
# 2. 特征与目标定义
# ----------------------
cat_features = ["Origin_Warehouse", "Destination", "Carrier"]
num_features = ["Weight_kg", "Cost", "Distance_miles"]
X = df[cat_features + num_features]
y_reg = df["Transit_Days"]  # 回归目标：实际时效

# ----------------------
# 3. 划分训练集和测试集
# ----------------------
X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.3, random_state=42)

# ----------------------
# 4. 预处理流水线
# ----------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ("num", StandardScaler(), num_features)
    ]
)

# ----------------------
# 5. 第一步：训练回归模型，预测基准时效
# ----------------------
reg_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", xgb.XGBRegressor(random_state=42))
])

reg_model.fit(X_train, y_train_reg)
y_pred_reg = reg_model.predict(X_test)
print("回归模型MAE:", mean_absolute_error(y_test_reg, y_pred_reg))


# ----------------------
# 6. 第二步：构造区间标签，训练分类模型预测+-2天内的概率
# ----------------------
# 构造训练集的区间标签
y_train_pred_reg = reg_model.predict(X_train)
train_lower = y_train_pred_reg - 2
train_upper = y_train_pred_reg + 2
y_train_cls = ((y_train_reg >= train_lower) & (y_train_reg <= train_upper)).astype(int)

# 构造测试集的区间标签
test_lower = y_pred_reg - 2
test_upper = y_pred_reg + 2
y_test_cls = ((y_test_reg >= test_lower) & (y_test_reg <= test_upper)).astype(int)

# 训练分类模型
cls_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="logloss"))
])

cls_model.fit(X_train, y_train_cls)
y_pred_cls_proba = cls_model.predict_proba(X_test)[:, 1]
print("分类模型AUC:", roc_auc_score(y_test_cls, y_pred_cls_proba))

# ----------------------
# 7. 示例：对一个新订单做预测
# ----------------------
new_order = pd.DataFrame({
    "Origin_Warehouse": ["Warehouse_MIA"],
    "Destination": ["San Francisco"],
    "Carrier": ["UPS"],
    "Weight_kg": [25.7],
    "Cost": [67.46],
    "Distance_miles": [291]
})

pred_days = reg_model.predict(new_order)[0]
prob_in_range = cls_model.predict_proba(new_order)[:, 1][0]

print(f"预测基准时效：{pred_days:.1f} 天")
print(f"时效落在 [{pred_days-2:.1f}, {pred_days+2:.1f}] 天内的概率：{prob_in_range:.2%}")