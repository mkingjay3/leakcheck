import pandas as pd

df = pd.read_csv("prices.csv")

# rolling window is built first, so mean() only ever looks backward from
# whatever row the window currently sits on
df['ma'] = df['close'].rolling(20).mean()
signal = df['close'] > df['ma']
