# real shape from QuantResearch - stat computed after the backtest,
# never feeds a decision. analyzer flags it, shouldn't.
import numpy as np

def stats(returns):
    return np.mean(returns) / np.std(returns)
