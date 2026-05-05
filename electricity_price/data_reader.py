import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 👇 解决中文乱码 + 负号显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Zen Hei', 'PingFang SC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 读取电价 =====================
def load_electricity_data(folder_path):
    df_list = []

    # 遍历文件夹里所有 xlsx 文件
    for file in os.listdir(folder_path):
        if file.endswith(".xlsx") and "现货出清数据" in file:
            print(f"正在读取：{file}")

            file_full_path = os.path.join(folder_path, file)

            # ✅ 关键：跳过前6行，从第7行开始读真正数据
            df = pd.read_excel(file_full_path, skiprows=range(1,6),header=0)

            # 提取需要的三列
            df = df[["时间", "日前价格", "实时价格"]].copy()

            df_list.append(df)

    # 合并所有文件
    df_total = pd.concat(df_list, ignore_index=True)

    # ==============================
    # ✅ 正确修复 24:00 时间格式
    # 例如把 2024-10-01 24:00 → 2024-10-02 00:00
    # ==============================
    def fix_time(time_str):
        time_str = str(time_str).strip()
        if "24:00" in time_str:
            date_part = time_str.split(" ")[0]
            new_date = pd.to_datetime(date_part) + pd.Timedelta(days=1)
            return new_date.strftime("%Y-%m-%d 00:00")
        return time_str

    df_total["时间"] = df_total["时间"].apply(fix_time)
    df_total["时间"] = pd.to_datetime(df_total["时间"])

    # 排序
    df_total = df_total.sort_values("时间").reset_index(drop=True)

    print("\n✅ 合并完成！")
    print(df.head())
    print(f"\n总数据条数：{len(df)}")

    return df_total


def plot_price_curve(df):
    plt.figure(figsize=(16, 6))

    # 画两条曲线
    plt.plot(df["时间"], df["日前价格"], label="日前价格", color="blue", linewidth=1.2)
    plt.plot(df["时间"], df["实时价格"], label="实时价格", color="red", linewidth=1.2, alpha=0.8)

    # 美化：时间轴格式
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))  # 每天一个主刻度
    plt.gcf().autofmt_xdate()  # 自动旋转日期标签

    plt.title("日前价格 vs 实时价格 时序曲线", fontsize=14)
    plt.xlabel("时间", fontsize=12)
    plt.ylabel("价格", fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ===================== 主函数 =====================
if __name__ == "__main__":
    # Excel 的文件夹路径
    folder = r"/Users/yukaifeng/Codes/Python/trail02/electricity_data"

    df = load_electricity_data(folder)
    plot_price_curve(df)

