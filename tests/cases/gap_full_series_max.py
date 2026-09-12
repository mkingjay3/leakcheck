import pandas as pd

df = pd.read_csv("prices.csv")
high_water = df['close'].max()
df['pct_of_high'] = df['close'] / high_water
