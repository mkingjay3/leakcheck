# real shape from Monte Carlo backtest.py:225 - a scalar summary. returns
# never appears outside the aggregate calls, so nothing is broadcast back
# over the series
import numpy as np

returns = load_returns()

drift = returns.mean() - returns.var() / 2
scaled = drift * 252
print(scaled)
