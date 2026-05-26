import pandas as pd
from data_reader import (load_electricity_clearing_data, load_electricity_bidding_space_data, load_weather_data,
                         genrate_env_flag)
import holidays
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

def generate_forward_feature(df_1, df_2, df_3):
    # =======================处理竞价空间数据============================
    # 初始化新列，构建竞价空间特征数据
    for i in range(8):
        df_3[f'日前竞价空间{i}'] = pd.NA
        df_3[f'统调负荷预测{i}'] = pd.NA
        df_3[f'外来电计划{i}'] = pd.NA
        df_3[f'光伏出力预测{i}'] = pd.NA
        df_3[f'风电出力预测{i}'] = pd.NA
        df_3[f'固定出力计划{i}'] = pd.NA
    # 遍历出清时间点，填充竞价空间数据
    for idx, target_time in df_3['时间_dt'].items():
        # 筛选时间点前两小时内的竞价数据
        mask = (df_1['时间_dt'] > target_time - pd.Timedelta(hours=2)) & (df_1['时间_dt'] <= target_time)
        window_data = df_1.loc[mask, ['时间_dt', '日前竞价空间', '统调负荷预测', '外来电计划', '光伏出力预测', '风电出力预测', '固定出力计划']].copy()
        # 按时间从近到远倒序排序，这样第一个元素就是距离target_time最近的
        window_data = window_data.sort_values('时间_dt', ascending=False).reset_index(drop=True)
        # 取往前两小时内的数据，不足的保留NA
        for i in range(min(8, len(window_data))):
            df_3.at[idx, f'日前竞价空间{i}'] = window_data['日前竞价空间'].iloc[i]
            df_3.at[idx, f'统调负荷预测{i}'] = window_data['统调负荷预测'].iloc[i]
            df_3.at[idx, f'外来电计划{i}'] = window_data['外来电计划'].iloc[i]
            df_3.at[idx, f'光伏出力预测{i}'] = window_data['光伏出力预测'].iloc[i]
            df_3.at[idx, f'风电出力预测{i}'] = window_data['风电出力预测'].iloc[i]
            df_3.at[idx, f'固定出力计划{i}'] = window_data['固定出力计划'].iloc[i]
    # =======================处理竞价天气数据============================
    # 初始化新列，构建天气特征数据
    for i in range(2):
        df_3[f'温度{i}'] = pd.NA
        df_3[f'湿度{i}'] = pd.NA
        df_3[f'降雨{i}'] = pd.NA
        df_3[f'辐照{i}'] = pd.NA
        df_3[f'云{i}'] = pd.NA
        df_3[f'风速{i}'] = pd.NA

    # 遍历出清时间点，填充天气数据
    for idx, target_time in df_3['时间_dt'].items():
        # 筛选时间点前两小时内的天气数据
        mask = (df_2['时间_dt'] > target_time - pd.Timedelta(hours=2)) & (df_2['时间_dt'] <= target_time)
        window_data = df_2.loc[mask, ['时间_dt', '温度', '湿度', '降雨', '辐照', '云', '风速']]

        # 按时间从近到远倒序排序，这样第一个元素就是距离target_time最近的
        window_data = window_data.sort_values('时间_dt', ascending=False).reset_index(drop=True)

        # 往前取两小时内的数据，不足的保留NA
        for i in range(min(2, len(window_data))):
            df_3.at[idx, f'温度{i}'] = window_data['温度'].iloc[i]
            df_3.at[idx, f'湿度{i}'] = window_data['湿度'].iloc[i]
            df_3.at[idx, f'降雨{i}'] = window_data['降雨'].iloc[i]
            df_3.at[idx, f'辐照{i}'] = window_data['辐照'].iloc[i]
            df_3.at[idx, f'云{i}'] = window_data['云'].iloc[i]
            df_3.at[idx, f'风速{i}'] = window_data['风速'].iloc[i]
    return df_3

def generate_that_day_feature(df_1, df_2, df_3):
    # =======================处理竞价空间数据============================
    # 初始化新列，构建竞价空间特征数据
    for i in range(4*24):
        df_3[f'day_日前竞价空间{i}'] = pd.NA
        df_3[f'day_统调负荷预测{i}'] = pd.NA
        df_3[f'day_外来电计划{i}'] = pd.NA
        df_3[f'day_光伏出力预测{i}'] = pd.NA
        df_3[f'day_风电出力预测{i}'] = pd.NA
        df_3[f'day_固定出力计划{i}'] = pd.NA
    # 遍历出清时间点，填充竞价空间数据
    for idx, target_time in df_3['时间_dt'].items():
        # 筛选当天的竞价数据
        mask = (df_1['时间_dt'].dt.date==target_time.date())
        window_data = df_1.loc[mask, ['时间_dt', '日前竞价空间', '统调负荷预测', '外来电计划', '光伏出力预测', '风电出力预测', '固定出力计划']].copy()
        # 当天数据不全的，直接跳过
        if len(window_data) != 4*24:
            continue
        for i in range(4*24):
            df_3.at[idx, f'day_日前竞价空间{i}'] = window_data['日前竞价空间'].iloc[i]
            df_3.at[idx, f'day_统调负荷预测{i}'] = window_data['统调负荷预测'].iloc[i]
            df_3.at[idx, f'day_外来电计划{i}'] = window_data['外来电计划'].iloc[i]
            df_3.at[idx, f'day_光伏出力预测{i}'] = window_data['光伏出力预测'].iloc[i]
            df_3.at[idx, f'day_风电出力预测{i}'] = window_data['风电出力预测'].iloc[i]
            df_3.at[idx, f'day_固定出力计划{i}'] = window_data['固定出力计划'].iloc[i]
    # =======================处理竞价天气数据============================
    # 初始化新列，构建天气特征数据
    for i in range(24):
        df_3[f'day_温度{i}'] = pd.NA
        df_3[f'day_湿度{i}'] = pd.NA
        df_3[f'day_降雨{i}'] = pd.NA
        df_3[f'day_辐照{i}'] = pd.NA
        df_3[f'day_云{i}'] = pd.NA
        df_3[f'day_风速{i}'] = pd.NA

    # 遍历出清时间点，填充天气数据
    for idx, target_time in df_3['时间_dt'].items():
        # 筛选时间点前两小时内的天气数据
        mask = (df_2['时间_dt'].dt.date==target_time.date())
        window_data = df_2.loc[mask, ['时间_dt', '温度', '湿度', '降雨', '辐照', '云', '风速']]
        if len(window_data) != 24:
            continue
        # 往前取两小时内的数据，不足的保留NA
        for i in range(24):
            df_3.at[idx, f'day_温度{i}'] = window_data['温度'].iloc[i]
            df_3.at[idx, f'day_湿度{i}'] = window_data['湿度'].iloc[i]
            df_3.at[idx, f'day_降雨{i}'] = window_data['降雨'].iloc[i]
            df_3.at[idx, f'day_辐照{i}'] = window_data['辐照'].iloc[i]
            df_3.at[idx, f'day_云{i}'] = window_data['云'].iloc[i]
            df_3.at[idx, f'day_风速{i}'] = window_data['风速'].iloc[i]
    return df_3

def generate_recent_prices_spread(df_3):
    # 先按时间排序
    df_3 = df_3.sort_values("时间").reset_index(drop=True)
    # 提取时间中的「时刻」部分（只保留时:分），用来匹配前几天的同一时间
    df_3["时刻"] = df_3["时间"].dt.strftime("%H:%M")
    #  循环生成前 1~7 天同一时间的价差
    for days in range(2, 8):
        col_name = f"价差_前{days}天"
        # 复制一份数据，把时间往前推 days 天，然后按「时间」+「时刻」合并，就能拿到同一时刻的历史数据
        tmp = df_3[["时间", "时刻", "价差（实时-日前）"]].copy()
        tmp["时间_偏移"] = tmp["时间"] + pd.Timedelta(days=days)
        # 合并：用「偏移后的时间」和「原时间」做 key，匹配同一时刻的历史价差
        merged = pd.merge(
            df_3,
            tmp[["时间_偏移", "时刻", "价差（实时-日前）"]].rename(columns={"价差（实时-日前）": col_name}),
            left_on=["时间", "时刻"],
            right_on=["时间_偏移", "时刻"],
            how="left"
        )
        # 只保留新增的那一列
        df_3[col_name] = merged[col_name]
    # 删掉辅助列
    df_3 = df_3.drop(columns=["时刻"])
    return df_3


def generate_feature_df(bidding_space_df: pd.DataFrame, weather_df: pd.DataFrame, clearing_df: pd.DataFrame, start_date, end_date)->pd.DataFrame:
    # 转时间格式
    df_1 = bidding_space_df.copy()
    df_1['时间_dt'] = pd.to_datetime(df_1['时间'])
    df_2 = weather_df.copy()
    df_2['时间_dt'] = pd.to_datetime(df_2['时间'])
    df_3 = clearing_df.copy()
    df_3['时间_dt'] = pd.to_datetime(df_3['时间'])

    df_3 = generate_recent_prices_spread(df_3)
    df_3 = generate_forward_feature(df_1, df_2, df_3)
    df_3 = generate_that_day_feature(df_1, df_2, df_3)


    df_3['日期']= df_3['时间'].dt.date
    df_3 = df_3[(df_3['日期'] >= pd.to_datetime(start_date).date()) & (df_3['日期'] <= pd.to_datetime(end_date).date())]

    # 电网工况
    df_3['grid_env'] = df_3['时间'].apply(genrate_env_flag)

    # 星期几（0=周一，6=周日）
    df_3['星期'] = df_3['时间_dt'].dt.weekday  # 0-6

    # 第几季度
    df_3['季度'] = df_3['时间_dt'].dt.quarter  # 1-4

    # 时间
    df_3['clock'] = (((df_3['时间_dt'].dt.hour) + (df_3['时间_dt'].dt.minute) / 60.0) * 2.0).astype(int)

    # 是否节假日（中国节假日，含周末）
    cn_holidays = holidays.China()
    df_3['是否节假日'] = df_3['时间_dt'].dt.date.apply(lambda x: x in cn_holidays or x.weekday() >= 5).astype(int)
    df_3['是否工作日'] = 1-df_3['是否节假日']

    df_3 = df_3.drop(columns=['日期', '时间_dt'])

    return df_3

def validate_model(model, x, true_y):
    pred_y = model.predict(x)
    # 计算指标
    rmse = np.sqrt(mean_squared_error(true_y, pred_y))
    r2 = r2_score(true_y, pred_y)

    # ✅ 计算：真实值 和 预测值 正负号一致的比例（方向准确率）
    sign_match = (np.sign(true_y) == np.sign(pred_y)).sum()
    total = len(true_y)
    sign_acc = sign_match / total

    # 把 true_y (Series) 和 pred_y (ndarray) 合并成一个DataFrame
    compare_df = pd.DataFrame({
        'true_y': true_y.reset_index(drop=True),  # 去掉索引，对齐pred_y
        'pred_y': pred_y
    })

    # 加上一列，标记是否预测正确
    compare_df['correct'] = (compare_df['true_y'] == compare_df['pred_y']).astype(int)

    # 输出
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"正负一致性准确率: {sign_acc:.4f}  ({sign_match}/{total})")

    return rmse, r2, sign_acc, sign_match, total



if __name__ == '__main__':
    # start_date = '2024-05-02'
    # end_date = '2026-05-14'
    # folder = r"/Users/yukaifeng/Codes/Python/trail02/electricity_data"
    # clearing_df = load_electricity_clearing_data(folder)
    # bidding_space_df = load_electricity_bidding_space_data(folder)
    # weather_df = load_weather_data(folder)
    # bidding_space_df['grid_env'] = bidding_space_df['时间'].apply(genrate_env_flag)
    # uninted_df = generate_feature_df(bidding_space_df, weather_df, clearing_df, start_date, end_date)
    # uninted_df.to_pickle("uninted_df.pkl")
    # print(f"✅ 数据集已保存：uninted_df.pkl")

    # 加载保存的数据集
    df = pd.read_pickle("uninted_df.pkl")
    print(f"✅ 数据集已加载：uninted_df.pkl")
    # 删除临时字段
    if '时间_dt' in df.columns:
        df = df.drop('时间_dt', axis=1)

    # 取近期的数据去训练
    start_date = '2024-06-01'
    df = df[df['时间'].dt.date >= pd.to_datetime(start_date).date()]
    end_date = '2026-05-13'
    df = df[df['时间'].dt.date <= pd.to_datetime(end_date).date()]
    # # 删掉没有燃气的行
    # df = df[df['grid_env'] == 2]
    # 按照日期选择最近的数据为验证集
    split_validation_data = '2026-05-01'
    validation_df = df[df['时间'].dt.date>=pd.to_datetime(split_validation_data).date()]
    df = df[df['时间'].dt.date < pd.to_datetime(split_validation_data).date()]
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,  # 0.2 作为测试集
        random_state=42  # 固定随机种子，保证每次划分结果一致
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
    train_X = train_df.iloc[:, 4:]  # 第2列～最后：特征
    train_y = train_df.iloc[:, 3]  # 第1列：要预测的目标
    test_X = test_df.iloc[:, 4:]
    test_y = test_df.iloc[:, 3]
    validation_X = validation_df.iloc[:, 4:]
    validation_y = validation_df.iloc[:, 3]

    train_y = (train_y> 0 ).astype(int)
    test_y = (test_y> 0).astype(int)
    validation_y = (validation_y> 0).astype(int)

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
    validation_X[categorical_cols] = validation_X[categorical_cols].astype('category')

    # ===================== 【7】XGBoost模型（开启 category 支持） =====================
    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=10,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        enable_categorical=True,  # 关键：支持枚举类型
    )

    # model = xgb.XGBRegressor(
    #     n_estimators=500,
    #     max_depth=10,
    #     learning_rate=0.08,
    #     subsample=0.9,
    #     colsample_bytree=0.9,
    #     reg_alpha=1,  # 加正则，防过拟合
    #     reg_lambda=1,
    #     random_state=42,
    #     n_jobs=-1,
    #     enable_categorical=True
    # )

    model = xgb.XGBClassifier(
        n_estimators=200,  # 减少树数量
        max_depth=4,  # 大幅降低深度！核心修复
        learning_rate=0.06,  # 慢一点学习，更稳
        subsample=0.7,  # 随机采样70%数据 → 防过拟合
        colsample_bytree=0.7,  # 随机采样70%特征 → 防过拟合
        reg_alpha=1,  # 加大L1正则
        reg_lambda=1,  # 加大L2正则（更重要）
        min_child_weight=2,  # 叶子节点最小样本数，防止过细
        random_state=42,
        n_jobs=-1,
        enable_categorical=True,
        eval_metric="logloss"
    )
    # model = xgb.XGBClassifier(
    #     n_estimators=120,
    #     max_depth=4,  # 浅一点，更稳
    #     learning_rate=0.07,
    #     subsample=0.7,
    #     colsample_bytree=0.7,
    #     reg_alpha=2.5,  # 强一点正则
    #     reg_lambda=4.0,  # 强一点正则
    #     min_child_weight=2,
    #     random_state=42,
    #     n_jobs=-1,
    #     enable_categorical=True,
    #     eval_metric="logloss"
    # )

    # from sklearn.ensemble import VotingClassifier
    # import lightgbm as lgb
    # from catboost import CatBoostClassifier
    #
    # # 1. 先处理类别特征
    # cat_features = [col for col in train_X.columns if train_X[col].dtype.name == 'category']

    # # 2. 定义两个基础模型
    # model1 = lgb.LGBMClassifier(
    #     n_estimators=150, max_depth=5, random_state=42, verbose=-1
    # )
    # model2 = CatBoostClassifier(
    #     iterations=150, depth=5, verbose=0, random_state=42,
    #     cat_features=cat_features  # 关键修复
    # )
    #
    # # 3. 融合模型
    # model = VotingClassifier(
    #     estimators=[('lgb', model1), ('cat', model2)],
    #     voting='soft'
    # )

    # 训练
    model.fit(train_X, train_y)

    print("=================训练集=================")
    rmse, r2, sign_acc, sign_match, total = validate_model(model, train_X, train_y)
    print("=================测试集=================")
    rmse, r2, sign_acc, sign_match, total = validate_model(model, test_X, test_y)
    print("=================验证集=================")
    rmse, r2, sign_acc, sign_match, total = validate_model(model, validation_X, validation_y)

