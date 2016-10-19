import math
import datetime

import pandas
import numpy as np

import data

def get_estimator(symbol, start, end, window=30, clean=True):
    
    prices = data.get_data(symbol, start, end)

    rs = (1 / (4 * math.log(2))) * ((prices['High'] / prices['Low']).apply(np.log))**2

    def f(v):
        return math.sqrt(252 * v.mean())
    
    result = rs.rolling(window=window, center=False).apply(func=f)
    result[:window-1] = np.nan
    
    if clean:
        return result.dropna()
    else:
        return result