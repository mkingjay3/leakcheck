import pandas as pd

#purposefully introduce bias with this code
df = pd.read_csv("prices.csv")

#.bfill takes the next available non empty value and fills it in 
df['close'] = df['close'].bfill()

#rolling mean avarages starting at 20 values
df['ma'] = df['close'].rolling(20).mean()
signal = df['close'] > df['ma']

#negative shift 
df['next_close'] = df['close'].shift(-1)

#lag comes from a variable, but it's assigned once at the top level, so the
#finder resolves it to 3 and correctly stays quiet about a positive shift
lag = 3
df['lagged'] = df['close'].shift(lag)

#normalizing with whole-series min/max: every row's scaled value depends on
#rows that hadn't happened yet, including the highest and lowest to come
close = df['close']
df['norm'] = (close - close.min()) / (close.max() - close.min())
