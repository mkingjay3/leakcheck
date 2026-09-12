# real shape from freqtrade-strategies lookahead_bias/DevilStra.py, which
# that repo's own readme documents as planted lookahead bias. The whole-
# series min/max are broadcast back across the series, so every row's
# normalized value depends on rows that hadn't happened yet. The column
# write happens in the caller, one scope away, so the only evidence
# visible here is the broadcast itself.


def normalize(df):
    df = (df - df.min()) / (df.max() - df.min())
    return df
