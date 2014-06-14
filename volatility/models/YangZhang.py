import math
import datetime

import pandas
import numpy as np

import data

def get_estimator(ticker, start, end, window=30, clean=True):
    
    prices = data.get_data(ticker, start, end)
    
    log_ho = (prices['Adj High'] / prices['Adj Open']).apply(np.log)
    log_lo = (prices['Adj Low'] / prices['Adj Open']).apply(np.log)
    log_co = (prices['Adj Close'] / prices['Adj Open']).apply(np.log)
    
    log_oc = (prices['Adj Open'] / prices['Adj Close'].shift(1)).apply(np.log)
    log_oc_sq = log_oc**2
    
    log_cc = (prices['Close'] / prices['Close'].shift(1)).apply(np.log)
    log_cc_sq = log_cc**2
    
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    
    close_vol = pandas.rolling_sum(log_cc_sq, window=window) * (1.0 / (window - 1.0))
    open_vol = pandas.rolling_sum(log_oc_sq, window=window) * (1.0 / (window - 1.0))
    window_rs = pandas.rolling_sum(rs, window=window) * (1.0 / (window - 1.0))
    
    result = (open_vol + 0.164333 * close_vol + 0.835667 * window_rs).apply(np.sqrt) * math.sqrt(252)
    
    result[:window-1] = np.nan

    if clean:
        return result.dropna()
    else:
        return result