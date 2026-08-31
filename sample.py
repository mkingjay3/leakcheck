import pandas as pd

#purposefully introduce bias with this code
df = pd.read_csv("prices.csv")

#.bfill takes the next available non empty value and fills it in 
#allows for row displacement
df['close'] = df['close'].bfill()

#rolling mean avarages starting at 20 values
df['ma'] = df['close'].rolling(20).mean()
signal = df['close'] > df['ma']

#negative shift pulls tomorrow's close into today's row
df['next_close'] = df['close'].shift(-1)

#lag comes from a variable, finder can't tell the sign without running it
lag = 3
df['lagged'] = df['close'].shift(lag)