import math

import numpy as np


def get_estimator(price_data, window=30, trading_periods=252, clean=True):

    log_hl = (price_data['High'] / price_data['Low']).apply(np.log)
    log_cc = (price_data['Close'] / price_data['Close'].shift(1)).apply(np.log)

    rs = 0.5 * log_hl**2 - (2*math.log(2)-1) * log_cc**2
    
    def f(v):
        return (trading_periods * v.mean())**0.5
    
    result = rs.rolling(window=window, center=False).apply(func=f)
    
    if clean:
        return result.dropna()
    else:
        return result
