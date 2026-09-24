# real shape from tsfresh's feature_calculators.py - the broadcast happens,
# but np.sum collapses it straight back to a count, so no per-row value ever
# escapes. A summary, the same as fp_posthoc_sharpe.py
import numpy as np


def ratio_beyond_r_sigma(x, r):
    return np.sum(np.abs(x - np.mean(x)) > r * np.std(x)) / x.size
