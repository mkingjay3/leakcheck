"""A minimal Series carrying the handful of pandas methods the strategies use.

This exists so the whole repository still runs on the standard library. The
strategies beside it are written as ordinary pandas code, and finder.py reads
them as ordinary pandas code: it works off syntax and never imports what it
analyses, so it neither knows nor cares that `close` is this class at run
time. That is what lets one file be both the thing leakcheck flags and the
thing the backtest runs, with the flagged line and the inflated Sharpe coming
from the same source.

Missing values are None and behave like pandas' NaN: arithmetic involving one
gives None, and every comparison with one is False.
"""

import math


def _paired(left, right):
    if isinstance(right, Series):
        return right.values
    return [right] * len(left.values)


class Rolling:
    """A window over a Series, ending on the current row unless centered."""

    def __init__(self, series, window, center):
        self.series = series
        self.window = window
        self.center = center

    def _apply(self, reduce):
        values = self.series.values
        out = []
        for index in range(len(values)):
            if index + 1 < self.window:
                out.append(None)
                continue
            chunk = values[index + 1 - self.window:index + 1]
            out.append(None if any(v is None for v in chunk) else reduce(chunk))

        result = Series(out)
        if self.center:
            # A centered window is the trailing window moved back half its
            # length, which is exactly the relationship finder.py models when
            # it reports a reach of window // 2 and when it clears a
            # compensating .shift(window // 2).
            result = result.shift(-(self.window // 2))
        return result

    def min(self):
        return self._apply(min)

    def max(self):
        return self._apply(max)

    def mean(self):
        return self._apply(lambda chunk: sum(chunk) / len(chunk))


class Series:

    def __init__(self, values):
        self.values = list(values)

    def __len__(self):
        return len(self.values)

    def __iter__(self):
        return iter(self.values)

    def _arithmetic(self, other, operate):
        out = []
        for left, right in zip(self.values, _paired(self, other)):
            if left is None or right is None:
                out.append(None)
                continue
            try:
                out.append(operate(left, right))
            except ZeroDivisionError:
                out.append(None)
        return Series(out)

    def _compare(self, other, operate):
        # a comparison against a missing value is False, never an error, so a
        # gap in the data means "no position" rather than a crash
        out = []
        for left, right in zip(self.values, _paired(self, other)):
            out.append(False if left is None or right is None else operate(left, right))
        return Series(out)

    def __sub__(self, other):
        return self._arithmetic(other, lambda a, b: a - b)

    def __add__(self, other):
        return self._arithmetic(other, lambda a, b: a + b)

    def __mul__(self, other):
        return self._arithmetic(other, lambda a, b: a * b)

    def __truediv__(self, other):
        return self._arithmetic(other, lambda a, b: a / b)

    def __lt__(self, other):
        return self._compare(other, lambda a, b: a < b)

    def __gt__(self, other):
        return self._compare(other, lambda a, b: a > b)

    def rolling(self, window, center=False):
        return Rolling(self, window, center)

    def shift(self, periods=1):
        """Move values `periods` rows forward; a negative count moves them back."""
        size = len(self.values)
        out = [None] * size
        for index in range(size):
            source = index - periods
            if 0 <= source < size:
                out[index] = self.values[source]
        return Series(out)

    def ffill(self):
        out = list(self.values)
        carried = None
        for index in range(len(out)):
            if out[index] is None:
                out[index] = carried
            else:
                carried = out[index]
        return Series(out)

    def bfill(self):
        out = list(self.values)
        carried = None
        for index in range(len(out) - 1, -1, -1):
            if out[index] is None:
                out[index] = carried
            else:
                carried = out[index]
        return Series(out)

    def min(self):
        present = [v for v in self.values if v is not None]
        return min(present) if present else None

    def max(self):
        present = [v for v in self.values if v is not None]
        return max(present) if present else None

    def pct_change(self):
        out = [None]
        for previous, current in zip(self.values, self.values[1:]):
            if previous is None or current is None or previous == 0:
                out.append(None)
            else:
                out.append(current / previous - 1)
        return Series(out)

    def equals(self, other, tolerance=1e-9):
        """True if two series match row for row, missing values included."""
        if len(self) != len(other):
            return False
        for left, right in zip(self.values, other.values):
            if (left is None) != (right is None):
                return False
            if left is not None and not math.isclose(left, right, abs_tol=tolerance):
                return False
        return True
