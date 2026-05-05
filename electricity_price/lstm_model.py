import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, RepeatVector, TimeDistributed

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False


# 构造时序样本，适配多步预测
def create_sequences(data, input_steps, output_steps):
    X, y = [], []
    # 滑动窗口构建输入输出对，确保输入为15天数据，输出为D日13:30到D+4日全量数据
    for i in range(input_steps, len(data) - output_steps + 1):
        X.append(data[i - input_steps: i, 0])
        y.append(data[i: i + output_steps, 0])
    return np.array(X), np.array(y)


# 搭建LSTM模型，适配长输入、多步输出
def build_lstm_model(input_steps, output_steps):
    model = Sequential()
    # 编码器，捕捉15天输入序列的长期依赖关系
    model.add(LSTM(128, input_shape=(input_steps, 1), return_sequences=False))
    model.add(Dropout(0.2))
    # 重复向量，匹配输出序列长度
    model.add(RepeatVector(output_steps))
    # 解码器，生成多步预测结果
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(32)))
    model.add(TimeDistributed(Dense(1)))

    model.compile(optimizer='adam', loss='mse')
    return model


# 主训练函数，实现D日13:30至D+4日价格预测
def lstm_forecast_price(df, target_col="日前价格"):
    # 基础参数设置（时间步长调整为30分钟）
    time_interval = 30  # 数据时间间隔（分钟），改为30min
    input_days = 15  # 输入时间长度（15天）
    predict_days = 4  # 预测天数（D+1至D+4，共4天）
    # 计算输入、输出时间步（30分钟/步，1天=48步）
    input_steps = int(input_days * 24 * 60 / time_interval)
    # 输出步长：D日13:30到D日24:00 + D+1至D+4日全量数据（D日13:30后共4.5天）
    output_steps = int((4.5) * 24 * 60 / time_interval)

    # 数据预处理与归一化
    data = df[target_col].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    # 构建训练样本
    X, y = create_sequences(data_scaled, input_steps, output_steps)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    y = y.reshape(y.shape[0], y.shape[1], 1)

    # 划分训练集、测试集（时序数据不打乱）
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # 模型训练
    model = build_lstm_model(input_steps, output_steps)
    model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )

    # 保存模型（路径可自行修改）
    joblib.dump(model, "price_model_epoch_20_batch_32_input15days_predict_D_D_4.pkl")
    # 保存缩放器（用于后续加载后反归一化）
    joblib.dump(scaler, "price_scaler_epoch_20_batch_32_input15days_predict_D_D_4.pkl")
    # 训练数据
    joblib.dump(X_train, "X_train_epoch_20_batch_32_input15days_predict_D_D_4.pkl")
    joblib.dump(y_train, "y_train_epoch_20_batch_32_input15days_predict_D_D_4.pkl")
    # 测试数据
    joblib.dump(X_test, "X_test_epoch_20_batch_32_input15days_predict_D_D_4.pkl")
    joblib.dump(y_test, "y_test_epoch_20_batch_32_input15days_predict_D_D_4.pkl")

    # 模型预测
    y_pred = model.predict(X_test, verbose=0)

    # 反归一化，还原真实价格
    y_test_flat = y_test.reshape(-1, 1)
    y_pred_flat = y_pred.reshape(-1, 1)
    y_test_real = scaler.inverse_transform(y_test_flat).reshape(y_test.shape)
    y_pred_real = scaler.inverse_transform(y_pred_flat).reshape(y_pred.shape)

    # 绘制预测对比图
    plt.figure(figsize=(16, 6))
    plt.plot(y_test_real[0, :, 0], label="真实价格", linewidth=2)
    plt.plot(y_pred_real[0, :, 0], label="LSTM预测价格", linewidth=2, alpha=0.8)
    plt.title(f"日前价格预测（D日13:30至D+4日）", fontsize=14)
    plt.xlabel("时间步（30分钟/步）", fontsize=12)
    plt.ylabel("价格", fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    return y_test_real, y_pred_real, model, scaler
