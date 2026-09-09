import pandas as pd

df = pd.read_csv("prices.csv")
lag = 3
df['lagged'] = df['close'].shift(lag)
