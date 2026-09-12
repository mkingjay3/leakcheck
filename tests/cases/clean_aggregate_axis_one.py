# real shape from QuantResearch/backtest/portfolio_optimization.py -
# mean(axis=1) averages the assets within each row, not down the time
# axis, so no row can pull from another
import pandas as pd

bm_ret = pd.read_csv("returns.csv")
bm_ret['benchmark'] = bm_ret.mean(axis=1)
