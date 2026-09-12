import pandas as pd

df = pd.read_csv("prices.csv")
vol = df['returns'].std()
df['zscore'] = df['returns'] / vol
