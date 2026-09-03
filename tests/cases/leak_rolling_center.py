import pandas as pd

df = pd.read_csv("prices.csv")
df['ma'] = df['close'].rolling(window=10, center=True).mean()
