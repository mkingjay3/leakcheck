# real shape from Oil Money CAD.py - np.std(m.resid) used to draw a
# sigma band on a matplotlib plot, not fed into a trading decision
import numpy as np

ax.fill_between(
    y_test.index,
    forecast + np.std(m.resid),
    forecast - np.std(m.resid),
)
