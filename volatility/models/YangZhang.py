import math

import numpy as np
import pandas as pd

from volatility.overlapping_sample import get_variance_overlapping_adjustment_factor

def get_estimator(price_data, window=30, trading_periods=252, clean=True, use_overlapping_adjustment_factor=True):

    m = get_variance_overlapping_adjustment_factor(window, len(price_data)) if use_overlapping_adjustment_factor else 1

    d = {}

    # open
    # NOTE: (open / yesterday's open) is how the function appears in Euan Sinclair's Volatility Trading, 2nd Edition, Equation 2.17b
    # d['log_oo'] = np.log(price_data['Open']     / price_data['Open'].shift(1))
    # This version of the equation makes more sense: https://portfolioslab.com/tools/yang-zhang
    d['log_oo'] = np.log(price_data['Open']     / price_data['Close'].shift(1))

    # close
    # NOTE: (close / yesterday's close) is how the function appears in Euan Sinclair's Volatility Trading, 2nd Edition, Equation 2.17c
    # d['log_cc'] = np.log(price_data['Close']    / price_data['Close'].shift(1))
    # This version of the equation makes more sense: https://portfolioslab.com/tools/yang-zhang
    d['log_cc'] = np.log(price_data['Close']    / price_data['Open'])

    # Rogers and Satchell
    log_hc = np.log(price_data['High']      / price_data['Close'])
    log_ho = np.log(price_data['High']      / price_data['Open'])
    log_lc = np.log(price_data['Low']       / price_data['Close'])
    log_lo = np.log(price_data['Low']       / price_data['Open'])
    
    d['rs']     = (log_hc * log_ho) + (log_lc * log_lo)

    df = pd.DataFrame(d)

    df['log_oo_avg']        = df['log_oo'].rolling(window=window, center=False).apply(func=lambda v: v.mean())
    df['log_oo_avg_sq']     = (df['log_oo'] - df['log_oo_avg']) ** 2
    df['log_oo_sigma_sq']   = df['log_oo_avg_sq'].rolling(window=window, center=False).apply(func=lambda v: v.sum() / (window - 1))

    df['log_cc_avg']        = df['log_cc'].rolling(window=window, center=False).apply(func=lambda v: v.mean())
    df['log_cc_avg_sq']     = (df['log_cc'] - df['log_cc_avg']) ** 2
    df['log_cc_sigma_sq']   = df['log_cc_avg_sq'].rolling(window=window, center=False).apply(func=lambda v: v.sum() / (window - 1))

    df['rs_sigma_sq']       = df['rs'].rolling(window=window, center=False).apply(func=lambda v: v.mean())

    k = 0.34 / (1.34 + ((window + 1) / (window - 1)))

    df['variance']          = df['log_oo_sigma_sq'] + (k * df['log_cc_sigma_sq']) + ((1 - k) * df['rs_sigma_sq'])
    df['vol']               = np.sqrt(df['variance']) * math.sqrt(trading_periods) * math.sqrt(m)

    if clean:
        return df['vol'].dropna()
    else:
        return df['vol']
