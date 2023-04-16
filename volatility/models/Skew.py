import numpy as np


def get_estimator(price_data, window=30, clean=True, use_overlapping_adjustment_factor=True):

    # use_overlapping_adjustment_factor does not make sense here... but variable has to be in the API interface

    log_return = (price_data['Close'] / price_data['Close'].shift(1)).apply(np.log)
    
    result = log_return.rolling(
        window=window,
        center=False
    ).skew()

    if clean:
        return result.dropna()
    else:
        return result
