import pandas as pd

prices = pd.read_csv(r"data\raw\prices\raw_prices_9e194f22_2021-06-17_to_2026-06-17.csv")


print(prices["SPY"].iloc[0])
