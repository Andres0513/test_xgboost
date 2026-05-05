from data_reader import load_electricity_data, plot_price_curve
from lstm_model import lstm_forecast_price

def main():
    # Excel 的文件夹路径
    folder = r"/Users/yukaifeng/Codes/Python/trail02/electricity_data"
    # 读数据
    df = load_electricity_data(folder)
    # 曲线图
    # plot_price_curve(df)
    # 开始预测
    y_true, y_pred, model = lstm_forecast_price(df, target_col="日前价格")


if __name__ == '__main__':
    main()