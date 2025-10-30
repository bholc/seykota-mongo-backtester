import pandas as pd
from pandas_datareader import data as web

df = web.DataReader('AAPL', 'stooq', start='2024-01-01')  # free Stooq source
print(df.sort_index().tail())