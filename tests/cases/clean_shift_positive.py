import pandas as pd

df = pd.read_csv("prices.csv")
df['prev'] = df['close'].shift(1)
df['ma'] = df['close'].rolling(20).mean()
