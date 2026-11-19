"""The same leak as leak_whole_series_stat.py, parked in a variable first.

Identical arithmetic, identical result, and leakcheck does not catch it. This
is the known gap the README describes: the statistic is bound to a name on one
line and applied on another, and catching that needs to know whether a
subscript target is a DataFrame or a dict. The version that guessed produced
26 false positives and caught no real leak, so it was removed.

It is here so the gap has a number against it rather than only a description:
whatever this costs is what the tool is currently missing.
"""

WINDOW = 252


def depth(close):
    close = close.ffill()
    low = close.min()
    high = close.max()
    return (close - low) / (high - low)
