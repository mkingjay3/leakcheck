# known gap: this leak is out of scope for the current detector, see README.md
import pandas as pd

df = pd.read_csv("prices.csv")
vol = df['returns'].std()
df['zscore'] = df['returns'] / vol
