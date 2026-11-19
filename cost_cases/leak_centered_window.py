"""clean.py, but the range windows are centered on the current row.

The one change is `center=True`, so each 252-day window straddles the row it
labels and half of it is future data. Reach is 126 rows.
"""

WINDOW = 252


def depth(close):
    close = close.ffill()
    low = close.rolling(WINDOW, center=True).min()
    high = close.rolling(WINDOW, center=True).max()
    return (close - low) / (high - low)
