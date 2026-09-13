# real shape from ta/trend.py (TRIX, KST, Ichimoku, DPO, Vortex) - the
# fill_value puts the whole-series mean into the rows the shift empties, so
# those rows depend on data from the whole history
import pandas as pd

df = pd.read_csv("prices.csv")
close = df['close']
df['prev'] = close.shift(1, fill_value=close.mean())
