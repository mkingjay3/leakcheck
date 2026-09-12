# the whole-series mean gates a branch directly, without ever being
# written into a column - a decision made against a number computed
# from rows that hadn't happened yet
import pandas as pd

df = pd.read_csv("prices.csv")
price = df['close'].iloc[-1]

if price > df['close'].mean():
    position = 1
