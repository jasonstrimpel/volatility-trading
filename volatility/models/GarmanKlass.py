import math

import numpy as np
import pandas as pd

from volatility.overlapping_sample import get_variance_overlapping_adjustment_factor


def get_estimator(price_data, window=30, trading_periods=252, clean=True, use_overlapping_adjustment_factor=True):

    m = get_variance_overlapping_adjustment_factor(window, len(price_data)) if use_overlapping_adjustment_factor else 1

    df = pd.DataFrame({
        'log_hl': (np.log(price_data['High']     / price_data['Low'])) ** 2

    # NOTE: (close / yesterday's close) is how the function appears in Euan Sinclair's Volatility Trading, 2nd Edition, Equation 2.15
    # and also in the original Garman-Klass paper (equation 19a) https://www.cmegroup.com/trading/fx/files/a_estimation_of_security_price.pdf
    # ,   'log_cX': (np.log(price_data['Close']    / price_data['Close'].shift(1))) ** 2
    # This version of the equation seem to produce "better looking" results  using the MSFT example data: https://portfolioslab.com/tools/garman-klass
    ,   'log_cX': (np.log(price_data['Close']    / price_data['Open'])) ** 2
    })

    df['term1'] = df['log_hl'].rolling(window=window, center=False).apply(func=lambda v: (v / 2).mean())
    df['term2'] = df['log_cX'].rolling(window=window, center=False).apply(func=lambda v: (((2 * math.log(2)) - 1) * v).mean())

    # NOTE: Sometimes `term1` - `term2` is negative - though this is not found in the literature... if this happens set to 0
    df['term_difference'] = df['term1'] - df['term2']
    df['term_difference'] = df['term_difference'].apply(lambda x: 0 if x < 0 else x)

    df['gk'] = np.sqrt(df['term_difference']) * math.sqrt(trading_periods) * math.sqrt(m)

    if clean:
        return df['gk'].dropna()
    else:
        return df['gk']
