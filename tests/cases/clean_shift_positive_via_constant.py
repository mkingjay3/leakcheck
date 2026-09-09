# lag resolves to 3 via collect_top_level_assignments - positive, so no
# finding either way, but this used to be named clean_shift_unknown.py
# from before that resolution existed; renamed since the shift here
# isn't actually unknown to the tool anymore
import pandas as pd

df = pd.read_csv("prices.csv")
lag = 3
df['lagged'] = df['close'].shift(lag)
