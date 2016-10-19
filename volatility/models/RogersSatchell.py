import math
import datetime

import pandas
import numpy as np

import data

def get_estimator(symbol, start, end, window=30, clean=True):
    
    prices = data.get_data(symbol, start, end)
    
    log_ho = (prices['High'] / prices['Open']).apply(np.log)
    log_lo = (prices['Low'] / prices['Open']).apply(np.log)
    log_co = (prices['Close'] / prices['Open']).apply(np.log)
    
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)

    def f(v):
        return math.sqrt(252 * v.mean())
    
    result = rs.rolling(window=window, center=False).apply(func=f)
    result[:window-1] = np.nan
    
    if clean:
        return result.dropna()
    else:
        return result