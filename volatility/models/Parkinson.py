import math

import numpy as np

from volatility.overlapping_sample import get_variance_overlapping_adjustment_factor


def get_estimator(price_data, window=30, trading_periods=252, clean=True, use_overlapping_adjustment_factor=True):

    m = get_variance_overlapping_adjustment_factor(window, len(price_data)) if use_overlapping_adjustment_factor else 1

    log_h_l_sq = np.log(price_data['High'] / price_data['Low'])**2

    def f(d):
        return math.sqrt(d.mean() / (4 * math.log(2)))
    
    result = log_h_l_sq.rolling(
        window=window,
        center=False
    ).apply(func=f) * math.sqrt(trading_periods) * math.sqrt(m)
    
    if clean:
        return result.dropna()
    else:
        return result
