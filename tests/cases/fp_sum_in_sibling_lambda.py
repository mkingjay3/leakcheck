# real shape from QuantResearch/backtest/portfolio_optimization.py - the bare
# w that np.sum(w) would broadcast over belongs to the SIBLING lambda, a
# different w in a different scope. Searching the whole statement flagged
# this; broadcast_root stops at the innermost lambda instead.
import numpy as np

cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
        {'type': 'ineq', 'fun': lambda w: w})
