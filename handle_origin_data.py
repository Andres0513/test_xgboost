import os
import glob
import pandas as pd

def load_all_files_in_folder_by_keyword(folder: str, keyword: str) -> pd.DataFrame:
    # 查找所有 .xlsx 且 包含关键词 的文件
    excel_files = [
        f for f in glob.glob(os.path.join(folder, "*.xlsx"))
        if keyword in os.path.basename(f)
    ]

    # 读取所有文件
    df_list = []
    for file in excel_files:
        print("正在读取：", os.path.basename(file))
        df = pd.read_excel(file)
        df_list.append(df)

    # 合并所有的DataFrame
    df_final = pd.concat(df_list, ignore_index=True)
    df_final = df_final.sort_values(by='时间', ascending=True)

    return df_final

def postprocess_merged_clearing_data(merged_data: pd.DataFrame) -> pd.DataFrame:
    df = merged_data.copy()
    # 1. 除了第一列（时间），其他列整体下移一行（相当于把上一行的值“推”到下一行）
    cols_to_shift = df.columns[1:]  # 除了第一列
    df[cols_to_shift] = df[cols_to_shift].shift(1)
    # 2. 去掉第一行（因为下移后第一行变成NaN了）
    df = df.drop(index=df.index[0]).reset_index(drop=True)
    return df

def postprocess_merged_bidding_space_data(merged_data: pd.DataFrame) -> pd.DataFrame:
    df = merged_data.copy()
    # 1. 除了第一列（时间），其他列整体下移一行（相当于把上一行的值“推”到下一行）
    cols_to_shift = df.columns[1:]  # 除了第一列
    df[cols_to_shift] = df[cols_to_shift].shift(1)
    # 2. 去掉第一行（因为下移后第一行变成NaN了）
    df = df.drop(index=df.index[0]).reset_index(drop=True)
    # 3. 列名映射
    rename_map = {
        "日前外来电": "风电出力实际",
        "日前风电": "外来电计划",
        "日前光伏": "风电出力预测",
        "实时负荷": "光伏出力预测",
        "实时外来电": "统调负荷实际",
        "实时风电": "外来电计划"
    }
    # 重命名列
    df = df.rename(columns=rename_map)
    return df

def  postprocess_merged_weather_data(merged_data: pd.DataFrame) -> pd.DataFrame:
    df = merged_data.copy()
    # 1. 除了第一列（时间），其他列整体上移一行（相当于把上一行的值“推”到下一行）
    cols_to_shift = df.columns[1:]  # 除了第一列
    df[cols_to_shift] = df[cols_to_shift].shift(-1)
    # 2. 去掉最后一行
    df = df.drop(index=df.index[-1]).reset_index(drop=True)
    def fix_time(time_str):
        time_str = str(time_str).strip()
        if "24:00" in time_str:
            date_part = time_str.split(" ")[0]
            new_date = pd.to_datetime(date_part) + pd.Timedelta(days=1)
            return new_date.strftime("%Y-%m-%d 00:00")
        return time_str
    df['时间'] = df['时间'].apply(fix_time)
    return df

# ===================== 主函数 =====================
if __name__ == '__main__':
    # 自动获取当前脚本所在目录
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # 原始文件夹名
    ORIGIN_FOLDER = "origin_data"
    # 输出文件夹名
    TARGET_FOLDER = 'electricity_data'
    # 要匹配的关键词
    KEY_WORDS = ["出清", "天气", "竞价空间"]
    for key_word in KEY_WORDS:
        # 拼接路径：项目目录/目标文件夹
        search_path = os.path.join(BASE_DIR, ORIGIN_FOLDER)
        target_path = os.path.join(os.path.join(BASE_DIR, TARGET_FOLDER), key_word+'.xlsx')
        df_final = load_all_files_in_folder_by_keyword(search_path, key_word)
        # 原文件数据有问题，后处理下
        if key_word == "出清":
            df_final = postprocess_merged_clearing_data(df_final)
        if key_word == "竞价空间":
            df_final = postprocess_merged_bidding_space_data(df_final)
        if key_word == "天气":
            df_final = postprocess_merged_weather_data(df_final)
        df_final.to_excel(target_path, index=False)
