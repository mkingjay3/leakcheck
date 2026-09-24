# the same idiom, but the shift is smaller than the window reaches. A centered
# window of 11 sees 5 rows ahead and shifting by 2 only undoes part of that,
# so rows 3 to 5 ahead still leak in.
import pandas as pd

df = pd.read_csv("prices.csv")
df['res'] = df['high'].rolling(window=11, center=True).apply(is_resistance).shift(2)
