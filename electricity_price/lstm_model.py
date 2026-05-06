import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

plt.rcParams['font.sans-serif'] = ['SimHei', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False

# 构造单步预测样本（递归预测专用）
def create_sequences(data, input_steps):
    X, y, y_origin = [], [], []
    for i in range(input_steps, len(data)):
        X.append(data[i - input_steps:i, 0])
        y.append(data[i, 0])

    return np.array(X), np.array(y)

# 递归预测模型：每次只预测下1个点
def build_lstm_model(input_steps):
    model = Sequential()
    model.add(LSTM(128, input_shape=(input_steps, 1), return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse')
    return model

def lstm_forecast_price(df, target_col="日前价格"):
    time_interval = 30
    input_days = 15
    input_steps = int(input_days * 24 * 60 / time_interval)
    output_steps = int(4.5 * 24 * 60 / time_interval)

    data = df[target_col].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    X, y = create_sequences(data_scaled, input_steps)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = build_lstm_model(input_steps)
    model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )

    joblib.dump(model, "price_model_recursive.pkl")
    joblib.dump(scaler, "price_scaler_recursive.pkl")
    joblib.dump(X_train, "X_train_recursive.pkl")
    joblib.dump(y_train, "y_train_recursive.pkl")
    joblib.dump(X_test, "X_test_recursive.pkl")
    joblib.dump(y_test, "y_test_recursive.pkl")
    joblib.dump(output_steps, "output_steps_recursive.pkl")

    return model, scaler, output_steps