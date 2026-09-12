# known gap: this leak is out of scope for the current detector, see README.md
import pandas as pd

df = pd.read_csv("prices.csv")
high_water = df['close'].max()
df['pct_of_high'] = df['close'] / high_water
