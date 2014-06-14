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
    
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)

    def f(v):
        return math.sqrt(252 * v.mean())
    
    result = pandas.rolling_apply(rs, window, f)
    result[:window-1] = np.nan
    
    if clean:
        return result.dropna()
    else:
        return result