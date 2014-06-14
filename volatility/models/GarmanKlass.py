import math
import datetime

import pandas
import numpy as np

import data

def get_estimator(ticker, start, end, window=30, clean=True):
    
    prices = data.get_data(ticker, start, end)
    
    log_hl = (prices['Adj High'] / prices['Adj Low']).apply(np.log)
    log_co = (prices['Adj Close'] / prices['Adj Open']).apply(np.log)
    log_oc = (prices['Adj Open'] / prices['Adj Close']).apply(np.log)
    
    rs = log_oc**2 + 0.5 * log_hl**2 - 0.3862 * log_co**2

    def f(v):
        return math.sqrt(252 * v.mean())
    
    result = pandas.rolling_apply(rs, window, f)
    result[:window-1] = np.nan
    
    if clean:
        return result.dropna()
    else:
        return result