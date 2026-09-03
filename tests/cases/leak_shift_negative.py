import pandas as pd

df = pd.read_csv("prices.csv")
df['next_close'] = df['close'].shift(-1)
