import math
import datetime

import pandas
import numpy as np

import data

def get_estimator(symbol, start, end, window=30, clean=True):
    
    prices = data.get_data(symbol, start, end)
    
    log_hl = (prices['High'] / prices['Low']).apply(np.log)
    log_co = (prices['Close'] / prices['Open']).apply(np.log)
    #log_oc = (prices['Open'] / prices['Close']).apply(np.log)
    
    #rs = log_oc**2 + 0.5 * log_hl**2 - 0.3862 * log_co**2
	
    rs = 0.5 * log_hl**2 - (2*math.log(2)-1) * log_co**2
	
    def f(v):
        return math.sqrt(252 * v.mean())
    
    result = rs.rolling(window=window, center=False).apply(func=f)
    result[:window-1] = np.nan
    
    if clean:
        return result.dropna()
    else:
        return result