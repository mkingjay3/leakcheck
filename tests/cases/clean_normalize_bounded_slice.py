# the same normalization as leak_normalize_inline.py, but over a slice
# bounded to known history, so nothing later than the decision point
# reaches it
import pandas as pd

df = pd.read_csv("prices.csv")
window = df['close'].iloc[-30:]
df['scaled'] = (window - window.min()) / (window.max() - window.min())
