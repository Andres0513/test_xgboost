import pandas as pd
import numpy as np
from enum import Enum
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns



class similarity_method(Enum):
    COSINE = 'COSINE'
    MAE = 'MAE'
    RMSE = 'RMSE'
    PEARSON = 'pearson'

SIMILARITY_METHOD = similarity_method.RMSE

def transfer_data(df):
    tmp = df.copy()
    tmp['日期'] = pd.to_datetime(tmp['时间']).dt.date

    d = tmp[['时间', '日前价格', '实时价格', '价差（实时-日前）']].copy()
    d['时间'] = pd.to_datetime(d['时间'])
    d['日期'] = d['时间'].dt.date
    d['时刻'] = d['时间'].dt.time

    # ============ 按日期获取特征 ============
    days_cols = [col for col in tmp.columns if 'day' in col.lower()]
    df_feat = tmp[['日期'] + days_cols + ['grid_env', '星期', '季度', '是否节假日']].copy()
    feat = df_feat.groupby('日期').first()

    # 生成 3 个独立DF（每行=日期，每列=00:00~23:30）
    da = d.pivot_table(index='日期', columns='时刻', values='日前价格', aggfunc='first')
    rt = d.pivot_table(index='日期', columns='时刻', values='实时价格', aggfunc='first')
    spread = d.pivot_table(index='日期', columns='时刻', values='价差（实时-日前）', aggfunc='first')

    # =============只要任意一个表有 NA，整行删除================
    mask_da = da.isna().any(axis=1)
    mask_rt = rt.isna().any(axis=1)
    mask_spread = spread.isna().any(axis=1)
    mask_feat = feat.isna().any(axis=1)
    bad_rows = mask_da | mask_rt | mask_spread | mask_feat
    # 三张表 同时删除坏行
    da = da[~bad_rows]
    rt = rt[~bad_rows]
    spread = spread[~bad_rows]
    feat = feat[~bad_rows]

    return da, rt, spread, feat

def cal_true_similarity_score(da, rt, spread):
    res = []
    dates = spread.index.to_list()
    data_matrix = spread.values
    for i in range(0, len(data_matrix)):
        for j in range(i + 1, len(data_matrix)):
            values1 = data_matrix[i]
            values2 = data_matrix[j]
            if SIMILARITY_METHOD == similarity_method.COSINE:
                score = cosine_similarity(values1.reshape(1, -1), values2.reshape(1, -1))[0][0]
                score = (score+1)/2
            if SIMILARITY_METHOD == similarity_method.RMSE:
                score = np.sqrt(np.mean((values1 - values2) ** 2))
            if SIMILARITY_METHOD == similarity_method.PEARSON:
                score = pearsonr(values1, values2)
            res.append(
                {
                    'date1': dates[i],
                    'date2': dates[j],
                    'similarity_score': score
                }
            )

    res = pd.DataFrame(res)
    return res

def plot_two_days(spread_df, date_str1, date_str2):
    # 把字符串日期转成 date 对象（匹配你的索引）
    date1 = datetime.strptime(date_str1, '%Y-%m-%d').date()
    date2 = datetime.strptime(date_str2, '%Y-%m-%d').date()

    # 取出两天数据
    day1 = spread_df.loc[date1]
    day2 = spread_df.loc[date2]

    # ✅ 修复：把 time 对象转成字符串 00:00, 00:30...
    times = [t.strftime("%H:%M") for t in spread_df.columns]

    # 画图
    plt.figure(figsize=(14, 6))
    plt.plot(times, day1.values, marker='o', linewidth=2, label=date_str1)
    plt.plot(times, day2.values, marker='s', linewidth=2, label=date_str2)

    plt.title("两日价差曲线对比", fontsize=14)
    plt.xlabel("时间")
    plt.ylabel("价差")
    plt.xticks(rotation=90)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_spread_distribution(df):
    # 取出价差列，自动忽略 NaN
    spread_data = df['价差（实时-日前）'].dropna()
    quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    q_list = spread_data.quantile(quantiles)
    print("===== 价差（实时-日前） 分位数 =====")
    for q, val in q_list.items():
        print(f"{int(q * 100)} 分位数: {val:.4f}")

    plt.figure(figsize=(10, 6))

    # 1. 直方图 + 核密度曲线
    sns.histplot(spread_data, kde=True, bins=30, color='skyblue', edgecolor='black')

    plt.title("价差列的概率分布", fontsize=14)
    plt.xlabel("价差（实时-日前）", fontsize=12)
    plt.ylabel("频数 / 密度", fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # 加载保存的数据集
    df = pd.read_pickle("/Users/yukaifeng/Codes/Python/trail02/ElecPriceSpreadPred/uninted_df.pkl")
    print(f"✅ 数据集已加载：uninted_df.pkl")
    da, rt, spread, feat = transfer_data(df)
    # scores = cal_true_similarity_score(da, rt, spread)
    # scores.to_excel('scores.xlsx')
    plot_spread_distribution(df)
    # plot_two_days(spread, '2025-01-06', '2026-01-01')
    a = 0