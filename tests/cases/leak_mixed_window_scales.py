# real shape from quantstats/stats.py:973 and ta/momentum.py:462, found by
# recall.py rather than by reading - a whole-series statistic divided into a
# rolling one. The rolling side is a value per row, so the whole-series side
# gets broadcast across every one of them.
import pandas as pd

df = pd.read_csv("prices.csv")
returns = df['returns']
df['ratio'] = returns.mean() / returns.rolling(20).std()
