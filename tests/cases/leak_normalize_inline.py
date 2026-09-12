# real shape from freqtrade-strategies lookahead_bias/Zeus.py, which that
# repo's readme documents as planted lookahead bias. Same leak as
# leak_normalize_helper.py, written straight into the column instead of
# going through a helper.
import pandas as pd

df = pd.read_csv("prices.csv")
tib = df['trend_ichimoku_base']
df['trend_ichimoku_base'] = (tib - tib.min()) / (tib.max() - tib.min())
