import pandas as pd

df = pd.read_csv("prices.csv")
df['close'] = df['close'].backfill()
