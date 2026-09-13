# a shift(-1) that builds a training label. Deliberate, and structurally
# identical to the bug, so the only way to settle it is for a reader to say so.
import pandas as pd

df = pd.read_csv("prices.csv")
df['target'] = df['close'].shift(-1)  # leakcheck: ignore
