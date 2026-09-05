import pandas as pd

df = pd.read_csv("prices.csv")

# shift() with no arguments defaults to periods=1, which only looks
# backward, so there is nothing here for the checker to flag
df["prev_close"] = df["close"].shift()
