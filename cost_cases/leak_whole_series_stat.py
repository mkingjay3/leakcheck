"""clean.py, but the range is taken over the whole series.

The one change is `close.min()` and `close.max()` over every row instead of a
trailing window, written inline the way freqtrade's planted exercises write
it. This is the bug the project is really about, and the one in DevilStra,
GodStraNew and Zeus: on day one the strategy already knows the high and the
low of the next ten years, so "near the bottom of the range" means near a
bottom that has not arrived yet.
"""

WINDOW = 252


def depth(close):
    close = close.ffill()
    return (close - close.min()) / (close.max() - close.min())
