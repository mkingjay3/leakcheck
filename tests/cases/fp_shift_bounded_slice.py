# real shape from QuantResearch/backtest/turtle.py - df_hist is already
# bounded to known history at decision time, so shift(-1) reindexes
# within an already-past 15-row slice rather than pulling in data that
# didn't exist yet. analyzer can't see the slice bounds it, flags it
# anyway. (possibly a real formula bug, next-close vs prev-close, but
# that's not a temporal leak.)
import pandas as pd

df_hist = pd.read_csv("history.csv")
diff = (df_hist.Close.iloc[-15:].shift(-1) - df_hist.High.iloc[-15:-1]).dropna()
