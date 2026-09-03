import pandas as pd

df = pd.read_csv("prices.csv")
df['ma'] = df['close'].rolling(20).mean()
