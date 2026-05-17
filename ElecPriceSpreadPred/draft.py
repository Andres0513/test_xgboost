import pandas as pd

# 先确保两个 df 的时间列都是 datetime 类型
df_1['时间'] = pd.to_datetime(df_1['时间'])
df_3['时间_dt'] = pd.to_datetime(df_3['时间_dt'])

# 给 df_1 按时间排序，方便后面快速查找
df_1 = df_1.sort_values('时间').reset_index(drop=True)

# 初始化新列，先填 NaN
for i in range(8):
    df_3[f'日前竞价空间{i}'] = pd.NA
    df_3[f'统调负荷预测{i}'] = pd.NA

# 遍历 df_3 的每一行
for idx, target_time in df_3['时间_dt'].items():
    # 筛选条件：时间 >= (target_time - 2h) 且 时间 < target_time
    mask = (df_1['时间'] >= target_time - pd.Timedelta(hours=2)) & (df_1['时间'] < target_time)
    window_data = df_1.loc[mask, ['日前竞价空间', '统调负荷预测']].copy()

    # 按时间从近到远排序（倒序），这样第一个元素就是离 target_time 最近的
    window_data = window_data.sort_values(df_1.columns[df_1.columns.get_loc('时间')], ascending=False)

    # 取最多8条，不足8条后面会保留 NA
    for i in range(min(8, len(window_data))):
        df_3.at[idx, f'日前竞价空间{i}'] = window_data['日前竞价空间'].iloc[i]
        df_3.at[idx, f'统调负荷预测{i}'] = window_data['统调负荷预测'].iloc[i]