import joblib

# 加载数据
X_train = joblib.load("X_train.pkl")
y_train = joblib.load("y_train.pkl")
X_test = joblib.load("X_test.pkl")
y_test = joblib.load("y_test.pkl")

# 加载模型和缩放器
model = joblib.load("price_model_epoch_30_batch_32_input15days_predict_D_D_4.pkl")
scaler = joblib.load("price_scaler_epoch_30_batch_32_input15days_predict_D_D_4.pkl")

# 直接预测
y_pred = model.predict(X_test, verbose=0)