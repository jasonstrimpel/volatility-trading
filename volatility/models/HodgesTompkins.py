import math

import numpy as np


def get_estimator(price_data, window=30, clean=True):
    
    log_return = (price_data['Close'] / price_data['Close'].shift(1)).apply(np.log)

    vol = log_return.rolling(
        window=window,
        center=False
    ).std() * math.sqrt(252)
    adj_factor = math.sqrt(
        (1.0 / (1.0 - (window / (log_return.count() - (window - 1.0))) +
                (window**2 - 1.0)/(3.0 * (log_return.count() - (window - 1.0))**2)))
    )

    result = vol * adj_factor

    if clean:
        return result.dropna()
    else:
        return result
