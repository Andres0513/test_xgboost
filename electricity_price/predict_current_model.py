import joblib

# 加载数据
X_train = joblib.load("/Users/yukaifeng/Codes/Python/trail02/electricity_price/X_train_epoch_20_batch_32_input15days_predict_D_D_4.pkl")
y_train = joblib.load("/Users/yukaifeng/Codes/Python/trail02/electricity_price/y_train_epoch_20_batch_32_input15days_predict_D_D_4.pkl")
X_test = joblib.load("/Users/yukaifeng/Codes/Python/trail02/electricity_price/X_test_epoch_20_batch_32_input15days_predict_D_D_4.pkl")
y_test = joblib.load("/Users/yukaifeng/Codes/Python/trail02/electricity_price/y_test_epoch_20_batch_32_input15days_predict_D_D_4.pkl")

# 加载模型和缩放器
model = joblib.load("/Users/yukaifeng/Codes/Python/trail02/electricity_price/price_model_epoch_20_batch_32_input15days_predict_D_D_4.pkl")
scaler = joblib.load("/Users/yukaifeng/Codes/Python/trail02/electricity_price/price_scaler_epoch_20_batch_32_input15days_predict_D_D_4.pkl")

# 直接预测
y_pred = model.predict(X_test, verbose=0)

a = 0