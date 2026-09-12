# real shape from dynamic_breakout_ii - bounded by slicing rather than by
# .rolling(); is_windowed counts a literal slice as windowed too.
import numpy as np

vol = np.std(df_hist.Close[-lookback_days:])
