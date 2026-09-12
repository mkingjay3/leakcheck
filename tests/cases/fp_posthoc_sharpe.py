# real shape from QuantResearch - a Sharpe ratio built from two aggregates
# over the same series, with the series itself never touched directly. A
# summary, not a per-row value.
import numpy as np

def stats(returns):
    return np.mean(returns) / np.std(returns)
