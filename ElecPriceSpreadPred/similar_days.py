import pandas as pd
from data_reader import load_electricity_clearing_data, load_electricity_bidding_space_data, load_weather_data, genrate_env_flag
import numpy as np

def postprocess_data(input: pd.DataFrame, start_date, end_date) -> pd.DataFrame:
    df = input.copy()
    df['时间'] = pd.to_datetime(df['时间'])
    df['日期'] = df['时间'].dt.date
    df = df[(df['日期'] >= pd.to_datetime(start_date).date()) & (df['日期'] <= pd.to_datetime(end_date).date())]
    result = {}
    for day, g in df.groupby('日期'):
        g = g.sort_values('时间')
        result[day] = {c: g[c].values for c in df.columns if c not in ['时间', '日期']}
    return result

def cal_similarity_score(cur: list, target: list) -> float:
    if len(cur) != len(target):
        return 1e6
    cur_arr = np.array(cur, dtype = np.float64)
    target_arr = np.array(target, dtype = np.float64)

    # 差值平方和
    score= np.sqrt(np.sum((cur_arr - target_arr) ** 2) / (np.sum(cur_arr ** 2) + 1e-6))
    score = float(score)
    if np.isnan(score):
        score = 1e6
    return score


def cal_spred_similarity(cur: list, target: list) -> float:
    if len(cur) != len(target):
        return -1.0

    match_count = 0
    total = len(cur)

    for c, t in zip(cur, target):
        # 同正 或 同负 → 记1
        if (c > 0 and t > 0) or (c < 0 and t < 0):
            match_count += 1

    # 计算匹配比例
    return float(match_count / total)

# ===================== 对每一行向前找前10个最相似的行 =====================
def take_top_10_forward_similar_day(da_snippets: list, grid_env_list: list, spred_snippets: list):
    n = len(da_snippets)
    top_10_indices = []
    for i in range(n):
        cur = da_snippets[i]
        # 存储 (行号j, 相似得分)
        similarity_score_list = []
        for j in range(i):
            target = da_snippets[j]
            similarity_score = cal_similarity_score(cur, target)
            spred_score = cal_spred_similarity(spred_snippets[i], spred_snippets[j])
            if grid_env_list[i] != grid_env_list[j]:
                similarity_score_list.append((j, 1e3, spred_score))
            else:
                similarity_score_list.append((j, similarity_score, spred_score))

        # 按误差从小到大排序（最相似在前）
        # similarity_score_list = [(j, -1000 if s >= 1000 else s, p) for j, s, p in similarity_score_list]
        # similarity_score_list.sort(key=lambda x: x[1], reverse=True)
        similarity_score_list.sort(key=lambda x: x[1])
        # 取前10个行号
        top_indices = [(idx, round(score,2), round(spred, 2)) for idx, score, spred in similarity_score_list[:10]]

        #不足10个的用-1补足
        while (len(top_indices) < 10):
            top_indices.append((-1, 1e3))

        top_10_indices.append(top_indices)

    return top_10_indices


# --------------------- 输出到 Excel ---------------------
def save_similar_result_to_excel(top_10_indices, date_array, save_path="相似日期结果.xlsx"):
    # 构造 DataFrame
    df = pd.DataFrame(
        top_10_indices,
        columns=[f"最相似日期{i + 1}" for i in range(10)]
    )

    # 插入第一列：当前日期（字符串）
    df.insert(0, "当前日期", date_array)

    # 写入 Excel
    df.to_excel(save_path, index=False)
    print(f"✅ Excel 已保存：{save_path}")


if __name__ == '__main__':
    start_date = '2024-05-02'
    end_date = '2026-04-30'
    folder = r"/Users/yukaifeng/Codes/Python/trail02/electricity_data"
    clearing_df = load_electricity_clearing_data(folder)
    bidding_space_df = load_electricity_bidding_space_data(folder)
    weather_df = load_weather_data(folder)
    bidding_space_df['grid_env'] = bidding_space_df['时间'].apply(genrate_env_flag)

    bidding_space_dict = postprocess_data(bidding_space_df, start_date, end_date)
    clearing_dict = postprocess_data(clearing_df, start_date, end_date)

    # 日期数组（字符串格式，已排序）
    date_array = sorted(bidding_space_dict.keys())
    da_snippets = [bidding_space_dict[d]['日前竞价空间'] for d in sorted(bidding_space_dict)]
    grid_env_list = [bidding_space_dict[d]['grid_env'][0] for d in sorted(bidding_space_dict)]
    spred_snippets = [clearing_dict[d]['价差（实时-日前）'] for d in sorted(clearing_dict)]
    top_10_indices = take_top_10_forward_similar_day(da_snippets, grid_env_list, spred_snippets)

    # 保存成 Excel
    save_similar_result_to_excel(top_10_indices, date_array)
    a = 0
