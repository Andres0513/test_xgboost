from data_reader import load_electricity_data

import numpy as np
import matplotlib.pyplot as plt
import joblib
import pandas as pd

plt.rcParams['font.family'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 加载 =====================
X_test = joblib.load("X_test_recursive.pkl")
y_test = joblib.load("y_test_recursive.pkl")
model = joblib.load("price_model_recursive.pkl")
scaler = joblib.load("price_scaler_recursive.pkl")
output_steps = joblib.load("output_steps_recursive.pkl")

# ===================== 递归预测 =====================
def recursive_predict(model, input_seq, n_steps):
    predictions = []
    current_seq = input_seq.copy()

    for _ in range(n_steps):
        pred = model.predict(current_seq.reshape(1, -1, 1), verbose=0)
        predictions.append(pred[0, 0])
        current_seq = np.append(current_seq[1:], pred)
        a = 0

    return np.array(predictions)

# 对测试集第一条样本预测
select_idx = 0
input_seq = X_test[select_idx, :, 0]
pred_seq = recursive_predict(model, input_seq, output_steps)

# 反归一化
pred_seq = scaler.inverse_transform(pred_seq.reshape(-1, 1)).flatten()

# 真实未来序列（构造未来真实曲线）
def get_true_future_seq(df, start_idx, n_steps, target_col="日前价格"):
    return df[target_col].iloc[start_idx:start_idx + n_steps].values

# Excel 的文件夹路径
folder = r"/Users/yukaifeng/Codes/Python/trail02/electricity_data"
# 读数据
df = load_electricity_data(folder)

true_seq = get_true_future_seq(df,
                               start_idx=len(df) - len(X_test) + select_idx + 15*48,
                               n_steps=output_steps)

# ===================== 误差 =====================
mask = true_seq > 1e-6
mape = np.mean(np.abs((true_seq[mask] - pred_seq[mask]) / true_seq[mask])) * 100

print(f"第{select_idx}组")
print(f"MAPE: {mape:.2f}%")

# ===================== 画图 =====================
plt.figure(figsize=(16, 6))
plt.plot(true_seq, label="真实价格", linewidth=1.5)
plt.plot(pred_seq, label="递归预测价格", linewidth=1.5, alpha=0.8)
plt.title(f"递归预测 D日13:30 ~ D+4日")
plt.xlabel("时间步（30分钟）")
plt.ylabel("价格")
plt.legend()
plt.grid(alpha=0.3)
plt.show()