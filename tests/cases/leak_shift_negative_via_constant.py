import pandas as pd

df = pd.read_csv("prices.csv")
lag = -1
df['next_close'] = df['close'].shift(lag)
