import torch
import pandas as pd
import os
import onnxruntime as ort
import joblib
import numpy as np

pre_release_start_date = '2026-05-01'
pre_release_end_date = '2026-05-14'

# ===================== 加载预发测试用数据 =====================
def load_pre_release_data():
    df = pd.read_pickle("uninted_df.pkl")
    df = df[(df['时间'].dt.date >= pd.to_datetime(pre_release_start_date).date()) &
            (df['时间'].dt.date <= pd.to_datetime(pre_release_end_date).date())]
    print(f"✅ 预发测试数据集已加载")
    return df

models_to_test = ["model_best_62.onnx", "model_best_63.onnx", "model_best_67.onnx"]
input = load_pre_release_data()
saved_input = input.copy()
for col in input.columns:
    input[col] = pd.to_numeric(input[col], errors='coerce')
# 提取特征
x = input.iloc[:, 4:]
# 切换到 mature 目录
os.chdir("mature")
# 加载预处理对象，保证和训练时完全一致
preprocessor = joblib.load("preprocessor.pkl")
# 这里不能用fit_transform
processed_x = preprocessor.transform(x)
processed_x = processed_x.astype(np.float32)
# 保留想要的列
saved_input = saved_input.iloc[:, :4]
for model in models_to_test:
    # 加载onnx模型
    session = ort.InferenceSession(model)
    input_name = session.get_inputs()[0].name
    # 执行 ONNX 推理
    pred_prob = session.run(None, {input_name: processed_x})[0]
    # 结果后处理（Sigmoid 输出转分类，大于0.5为1，否则为0）
    pred_class = (pred_prob > 0.5).astype(int).flatten()
    saved_input[model + '_prediction'] = pred_class
    saved_input[model + '_prob'] = pred_prob
saved_input.to_excel('测试结果.xlsx')
print(f"✅ 预发测试结果统计完成: 测试结果.xlsx")