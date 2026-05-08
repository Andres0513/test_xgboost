import pandas as pd
from data_reader import load_electricity_clearing_data, load_electricity_bidding_space_data, load_weather_data

if __name__ == '__main__':
    folder = r"/Users/yukaifeng/Codes/Python/trail02/electricity_data"
    clearing_df = load_electricity_clearing_data(folder)
    bidding_space_df = load_electricity_bidding_space_data(folder)
    weather_df = load_weather_data(folder)
