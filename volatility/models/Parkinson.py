import math

import numpy as np


def get_estimator(price_data, window=30, clean=True):

    rs = (1 / (4 * math.log(2))) * ((price_data['High'] / price_data['Low']).apply(np.log))**2

    def f(v):
        return math.sqrt(252 * v.mean())
    
    result = rs.rolling(
        window=window,
        center=False
    ).apply(func=f)
    
    if clean:
        return result.dropna()
    else:
        return result
