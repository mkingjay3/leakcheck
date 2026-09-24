# real shape from NostalgiaForInfinity - a centered window of 5 reaches 2 rows
# ahead, and the .shift(2) puts every value back on a row whose window is
# entirely behind it. The standard support/resistance idiom, and correct.
import pandas as pd

df = pd.read_csv("prices.csv")
df['res'] = df['high'].rolling(window=5, center=True).apply(is_resistance).shift(2)
