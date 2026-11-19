"""clean.py, but gaps in the feed are filled backward instead of forward.

The one changed line is the fill. A missing bar takes the NEXT known price,
so on every gap day the strategy is handed a price that had not happened yet.
Reach is unbounded: the next valid value can be any distance ahead.
"""

WINDOW = 252


def depth(close):
    close = close.bfill()
    low = close.rolling(WINDOW).min()
    high = close.rolling(WINDOW).max()
    return (close - low) / (high - low)
