# real shape from Monte Carlo backtest.py:225 - the stat is assigned to
# a scalar that only feeds more scalars and a print, never a comparison
# or a column write, so there's no evidence it informs a decision
import numpy as np

returns = load_returns()

drift = returns.mean() - returns.var() / 2
scaled = drift * 252
print(scaled)
