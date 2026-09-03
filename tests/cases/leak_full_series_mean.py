import pandas as pd

df = pd.read_csv("prices.csv")
# one number computed over the whole series, used as a static threshold
df['threshold'] = df['close'].mean()
signal = df['close'] > df['threshold']
