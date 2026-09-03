import pandas as pd

df = pd.read_csv("prices.csv")
df['expanding_mean'] = df['close'].expanding().mean()
