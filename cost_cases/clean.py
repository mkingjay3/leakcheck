"""Where price sits inside its trailing 252-day range, honestly measured.

Everything read here is behind the row being decided: the window ends on that
row, so a value on day i uses days i-251 through i and nothing else. This is
the control the leaked variants are measured against, and each of them differs
from it by exactly one line.

Each variant returns `depth`: where the latest price sits inside its range,
0 at the low and 1 at the high. The entry rule is applied by cost.py, in both
directions, so no variant is credited or blamed for the direction it happens
to suit. `close` is anything carrying the pandas Series methods used below.
"""

WINDOW = 252


def depth(close):
    close = close.ffill()
    low = close.rolling(WINDOW).min()
    high = close.rolling(WINDOW).max()
    return (close - low) / (high - low)
