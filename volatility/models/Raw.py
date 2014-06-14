import math
import datetime

import pandas
import numpy as np

import data

def get_estimator(ticker, start, end, window=30, clean=True):
    
    prices = data.get_data(ticker, start, end)
    
    log_return = (prices['Adj Close'] / prices['Adj Close'].shift(1)).apply(np.log)

    result = pandas.rolling_std(log_return, window=window) * math.sqrt(252)
    
    result[:window-1] = np.nan
    
    if clean:
        return result.dropna()
    else:
        return result