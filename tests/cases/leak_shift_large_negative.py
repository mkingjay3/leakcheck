import pandas as pd

df = pd.read_csv("prices.csv")
df['future_close'] = df['close'].shift(-10)
