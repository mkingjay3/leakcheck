import pandas as pd

df = pd.read_csv("prices.csv")
total_volume = df['volume'].sum()
df['volume_weight'] = df['volume'] / total_volume
