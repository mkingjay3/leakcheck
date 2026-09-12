# real shape from quant-trading/Shooting Star backtest.py:42 - one mean over
# the whole frame decides, row by row, whether each candle's body is small,
# so every candle is judged against data from the whole backtest
import numpy as np

df = load_prices()
body = df['Open'] - df['Close']
df['small_body'] = np.where(abs(body) < abs(np.mean(body)) * 0.5, 1, 0)
