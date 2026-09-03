import pandas as pd

df = pd.read_csv("prices.csv")
df['prev3'] = df['close'].shift(periods=3)
