import math
import datetime

import pandas
import numpy as np

import data

def get_estimator(symbol, start, end, window=30, clean=True):
    
    prices = data.get_data(symbol, start, end)
    
    log_return = (prices['Close'] / prices['Close'].shift(1)).apply(np.log)
	
    result = log_return.rolling(window=window,center=False).skew()
    
    result[:window-1] = np.nan
    
    if clean:
        return result.dropna()
    else:
        return result