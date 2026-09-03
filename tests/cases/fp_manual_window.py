# real shape from dynamic_breakout_ii - windowed by slicing rather
# than .rolling(), so it's bounded but the analyzer can't see that.
import numpy as np

vol = np.std(df_hist.Close[-lookback_days:])
