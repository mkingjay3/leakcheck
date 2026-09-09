# real shape from Smart Farmers/country selection.py - groupby is a
# categorical aggregate (by country and year), not a time-ordered one
import pandas as pd

trade = pd.read_csv("trade.csv")
export = trade.groupby(['Area', 'Year']).sum()
