import pandas as pd
from data_reader import load_electricity_clearing_data, load_electricity_bidding_space_data, load_weather_data
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    folder = r"/Users/yukaifeng/Codes/Python/trail02/electricity_data"
    clearing_df = load_electricity_clearing_data(folder)
    bidding_space_df = load_electricity_bidding_space_data(folder)
    df = clearing_df
    start_date = '2024-05-02'
    end_date = '2026-04-30'
    df['时间'] = pd.to_datetime(df['时间'])
    df['日期'] = df['时间'].dt.date
    df = df[(df['日期'] >= pd.to_datetime(start_date).date()) & (df['日期'] <= pd.to_datetime(end_date).date())]

    result = {}
    for day, g in df.groupby('日期'):
        g = g.sort_values('时间')
        result[day] = {c: g[c].values for c in df.columns if c not in ['时间', '日期']}

    # 日期数组（字符串格式，已排序）
    date_array = sorted(result.keys())
    da_snippets = [result[d]['价差（实时-日前）'] for d in sorted(result)]
    # da_snippets = [np.where(result[d]['价差（实时-日前）']>0, 1, -1) for d in sorted(result)]

    # indices_to_plot = [532,	524, 442]
    indices_to_plot = [726,	714]
    indices_to_plot = [714, 706]
    indices_to_plot = [714, 136]
    indices_to_plot = [727, 721, 714, 720, 726, 704]

    # --------------------- 绘图代码（正式开始） ---------------------
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 解决 Mac 中文显示
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(14, 6))

    # 画出你指定的每一天曲线
    for idx in indices_to_plot:
        if idx < 0 or idx >= len(da_snippets):
            continue

        day_str = str(date_array[idx])
        curve = da_snippets[idx]
        time_points = np.arange(len(curve))  # 0~47 点

        plt.plot(
            time_points,
            curve,
            linewidth=2,
            label=f"日期：{day_str} (索引={idx})"
        )

    # 图表样式
    plt.title("日前竞价空间 - 每日48点曲线对比", fontsize=14)
    plt.xlabel("时段点 (0~47)", fontsize=12)
    plt.ylabel("日前竞价空间", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()

    # 显示图片
    plt.show()