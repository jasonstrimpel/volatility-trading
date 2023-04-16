import math

import numpy as np

from volatility.overlapping_sample import get_variance_overlapping_adjustment_factor

def get_estimator(price_data, window=30, trading_periods=252, clean=True, use_overlapping_adjustment_factor=True):

    m = get_variance_overlapping_adjustment_factor(window, len(price_data)) if use_overlapping_adjustment_factor else 1
    
    log_hc = np.log(price_data['High']  / price_data['Close'])
    log_ho = np.log(price_data['High']  / price_data['Open'])
    log_lc = np.log(price_data['Low']   / price_data['Close'])
    log_lo = np.log(price_data['Low']   / price_data['Open'])
    
    rs = (log_hc * log_ho) + (log_lc * log_lo)

    def f(v):
        return math.sqrt(v.mean())
    
    result = rs.rolling(
        window=window,
        center=False
    ).apply(func=f) * math.sqrt(trading_periods) * math.sqrt(m)
    
    if clean:
        return result.dropna()
    else:
        return result
