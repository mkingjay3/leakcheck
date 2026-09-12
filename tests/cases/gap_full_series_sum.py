# known gap: this leak is out of scope for the current detector, see README.md
import pandas as pd

df = pd.read_csv("prices.csv")
total_volume = df['volume'].sum()
df['volume_weight'] = df['volume'] / total_volume
