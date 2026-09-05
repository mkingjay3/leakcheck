# the slicing happens one line before the aggregate call, on a separate
# variable, so by the time std() runs there is no rolling/expanding/ewm
# in sight, even though the series was already bounded to 30 rows
import pandas as pd

df = pd.read_csv("prices.csv")

window = df["close"].iloc[-30:]
vol = window.std()
threshold = vol * 2
