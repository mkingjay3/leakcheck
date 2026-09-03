import pandas as pd

df = pd.read_csv("prices.csv")
df['ewm_mean'] = df['close'].ewm(span=10).mean()
