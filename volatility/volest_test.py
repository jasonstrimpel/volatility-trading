import os

import pandas as pd
import yfinance as yf

from volatility import volest
from volatility import data as data_helpers

def data_save(symbol, data):
    if not os.path.exists('tests/'):
        os.makedirs('tests/')
    data = data.reset_index()
    data['Date'] = pd.to_datetime(data['Date']).dt.date
    data.to_csv(f'tests/{symbol}.csv', index=False)


def test_overlapping_sample_msft():

    def get_yf_data(symbol):

        csv_path = f'tests/{symbol}.csv'

        if not os.path.exists(csv_path):
            data = yf.download([symbol], start="2003-05-21", end="2007-05-21")
            data_save(symbol, data)

        # use the yahoo helper to correctly format data from finance.yahoo.com
        return data_helpers.yahoo_helper(symbol, csv_path)

    for estimator in ['Raw', 'Parkinson', 'GarmanKlass', 'RogersSatchell', 'YangZhang']:

        # estimator windows
        # don't change these... they are specific to this test
        windows = [20, 40, 60, 120]
        quantiles = [0.25, 0.75]

        # other vars
        window = 30
        bins = 100
        normed = True

        use_overlapping_adjustment_factor = True

        # MSFT is used as the example in Volatility Trading, 2nd Edition
        subject_price_data  = get_yf_data('MSFT')
        bench_price_data    = get_yf_data('SPY')

        # initialize class
        vol = volest.VolatilityEstimator(
            price_data=subject_price_data,
            estimator=estimator,
            bench_data=bench_price_data
        )

        # call plt.show() on any of the below...
        _, plt, cone_datas = vol.cones(windows=windows, quantiles=quantiles
            # ,   use_overlapping_adjustment_factor=use_overlapping_adjustment_factor
        )

        # _, plt = vol.rolling_quantiles(window=window, quantiles=quantiles)
        # _, plt = vol.rolling_extremes(window=window)
        # _, plt = vol.rolling_descriptives(window=window)
        # _, plt = vol.histogram(window=window, bins=bins, normed=normed)

        # _, plt = vol.benchmark_compare(window=window)
        # _, plt = vol.benchmark_correlation(window=window)

        # if not os.path.exists('term-sheets/'):
        #     os.makedirs('term-sheets/')

        # # ... or create a pdf term sheet with all metrics in term-sheets/
        vol.term_sheet(
            window,
            windows,
            quantiles,
            bins,
            normed
        # ,   use_overlapping_adjustment_factor
        )

        if estimator == 'Raw' and False:

            window_datas = {}

            for i in range(len(windows)):

                window      = windows[i]
                cone_data   = cone_datas[i]

                window_datas[window] = {
                    'max':                          round(cone_data.max(), 3)
                ,   f'{int(quantiles[1] * 100)}%':  round(cone_data.quantile(quantiles[1]), 3)
                ,   'median':                       round(cone_data.median(), 3)
                ,   f'{int(quantiles[0] * 100)}%':  round(cone_data.quantile(quantiles[0]), 3)
                ,   'min':                          round(cone_data.min(), 3)
                # ,   realized.append(estimator[-1])
                }

            # from Volatilty Trading, 2nd Edition, Volatilty Forecasting, p59 in my hard back copy
            window_datas_expected = {}

            assert 20 == windows[0]
            window_datas_expected[windows[0]] = {
                'max':                          0.465
            ,   f'{int(quantiles[1] * 100)}%':  0.213
            ,   'median':                       0.159
            ,   f'{int(quantiles[0] * 100)}%':  0.123
            ,   'min':                          0.062
            }

            assert 40 == windows[1]
            window_datas_expected[windows[1]] = {
                'max':                          0.352
            ,   f'{int(quantiles[1] * 100)}%':  0.213
            ,   'median':                       0.172
            ,   f'{int(quantiles[0] * 100)}%':  0.149
            ,   'min':                          0.096
            }

            assert 60 == windows[2]
            window_datas_expected[windows[2]] = {
                'max':                          0.287
            ,   f'{int(quantiles[1] * 100)}%':  0.225
            ,   'median':                       0.169
            ,   f'{int(quantiles[0] * 100)}%':  0.147
            ,   'min':                          0.091
            }

            assert 120 == windows[3]
            window_datas_expected[windows[3]] = {
                'max':                          0.258
            ,   f'{int(quantiles[1] * 100)}%':  0.200
            ,   'median':                       0.181
            ,   f'{int(quantiles[0] * 100)}%':  0.161
            ,   'min':                          0.136
            }

            print(f"error results... ")

            error_total = 0

            for i in range(len(windows)):

                window = windows[i]

                window_data             = window_datas[window]
                window_data_expected    = window_datas_expected[window]

                error_window = 0

                for k in window_data.keys():

                    assert k in window_data_expected

                    error_window += abs(window_data[k] - window_data_expected[k])

                print(f"{'%5d' % (window, )}: {error_window}")

                error_total += error_window

            print(f"total: {error_total}")
