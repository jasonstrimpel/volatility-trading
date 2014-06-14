import math
import datetime

import pandas
import numpy as np

import data

def get_estimator(ticker, start, end, window=30, clean=True):
    
    prices = data.get_data(ticker, start, end)

    rs = (1 / (4 * math.log(2))) * ((prices['Adj High'] / prices['Adj Low']).apply(np.log))**2

    def f(v):
        return math.sqrt(252 * v.mean())
    
    result = pandas.rolling_apply(rs, window, f)
    result[:window-1] = np.nan
    
    if clean:
        return result.dropna()
    else:
        return result