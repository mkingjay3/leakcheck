"""clean.py, but the entry test reads tomorrow's close.

The one changed line is `close.shift(-1)`, the shape someone writes when they
mean a one-day lag and get the sign backwards. Reach is exactly one row, the
smallest leak of the four, which makes it the useful control on whether reach
predicts cost.
"""

WINDOW = 252


def depth(close):
    close = close.ffill()
    low = close.rolling(WINDOW).min()
    high = close.rolling(WINDOW).max()
    return (close.shift(-1) - low) / (high - low)
