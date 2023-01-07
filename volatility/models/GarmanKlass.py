import math

import numpy as np

from volatility.models.api import get_variance_overlapping_adjustment_factor


def get_estimator(price_data, window=30, trading_periods=252, clean=True, use_overlapping_adjustment_factor=True):

    m = get_variance_overlapping_adjustment_factor(window, len(price_data)) if use_overlapping_adjustment_factor else 1

    log_hl = (price_data['High'] / price_data['Low']).apply(np.log)
    log_co = (price_data['Close'] / price_data['Open']).apply(np.log)

    rs = 0.5 * log_hl**2 - (2*math.log(2)-1) * log_co**2
    
    def f(v):
        return ((trading_periods * v.mean())**0.5) * math.sqrt(m)
    
    result = rs.rolling(window=window, center=False).apply(func=f)
    
    if clean:
        return result.dropna()
    else:
        return result
