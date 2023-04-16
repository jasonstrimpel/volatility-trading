import datetime
import os

import pandas
import numpy
from scipy.stats import norm
import statsmodels.api as sm
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from volatility import models

ESTIMATORS = [
    'GarmanKlass',
    'Kurtosis',
    'Parkinson',
    'Raw',
    'RogersSatchell',
    'Skew',
    'YangZhang'
]
PRICE_COLUMNS = {
    'Open',
    'High',
    'Low',
    'Close'
}


def array_to_dataframe(ndarray):
    return pandas.DataFrame(
        ndarray,
        columns=['Open', 'High', 'Low', 'Close']
    )


class VolatilityEstimator(object):

    def __init__(self, price_data, estimator, bench_data=None):
        """Constructor for volatility estimators
        
        Parameters
        ----------
        price_data: pandas.DataFrame or numpy.ndarray
            If pandas.DataFrame, must include columns Open, High, Low, Close. Also
            must include property symbol with the symbol we're working with. If
            numpy.ndarray, must be of shape (r, 4) with columns in order of open,
            high, low, close prices. If numpy.ndarray, will be coerced to pandas.DataFrame
            with no date data
        estimator : string
            Estimator estimator; valid arguments are:
                "GarmanKlass", "Kurtosis", "Parkinson", "Raw",
                "RogersSatchell", "Skew", "YangZhang"
        """

        if not isinstance(price_data, numpy.ndarray) and not \
                isinstance(price_data, pandas.DataFrame):
            raise ValueError('price_data must be of type numpy.ndarray or pandas.DataFrame')
        if isinstance(price_data, numpy.ndarray) and price_data.shape[0] != 4:
            raise ValueError('price_data of type numpy.ndarray shape of (r, 4)')
        if isinstance(price_data, pandas.DataFrame) and not \
                PRICE_COLUMNS.issubset(price_data.columns):
            raise ValueError('price_data requires Open, High, Low, Close')
        if price_data.symbol is None or price_data.symbol == '':
            raise ValueError('Symbol required as property of price_data')
        if estimator not in ESTIMATORS:
            raise ValueError('Acceptable volatility model is required')

        if isinstance(price_data, numpy.ndarray):
            price_data = array_to_dataframe(price_data)
            price_data.symbol = '-NA-'
            start = price_data.index[0]
            end = price_data.index[0]
        else:
            start = price_data.index[0].to_pydatetime().strftime('%Y-%m-%d')
            end = price_data.index[-1].to_pydatetime().strftime('%Y-%m-%d')

        if bench_data is not None:
            if price_data.shape != bench_data.shape:
                raise ValueError('price_data and bench_data must be same shape')
            if not isinstance(bench_data, numpy.ndarray) and not \
                    isinstance(bench_data, pandas.DataFrame):
                raise ValueError('bench_data must be of type numpy.ndarray or pandas.DataFrame')
            if isinstance(bench_data, numpy.ndarray) and bench_data.shape[0] != 4:
                raise ValueError('bench_data of type numpy.ndarray shape of (r, 4)')
            if isinstance(bench_data, pandas.DataFrame) and not \
                    PRICE_COLUMNS.issubset(bench_data.columns):
                raise ValueError('bench_data requires Open, High, Low, Close')
            if bench_data.symbol is None or bench_data.symbol == '':
                raise ValueError('Symbol required as property of bench_data')

            # bench_data = bench_data.loc[start:end]

            if isinstance(bench_data, numpy.ndarray):
                bench_data = array_to_dataframe(bench_data)
                bench_data.symbol = '-NA-'

            self._bench_data = bench_data
            self._bench_symbol = bench_data.symbol

        self._price_data = price_data
        self._symbol = price_data.symbol
        self._start = start
        self._end = end
        self._estimator = estimator
        
        matplotlib.rc('image', origin='upper')

        matplotlib.rcParams['font.size'] = '11'
        
        matplotlib.rcParams['grid.color'] = 'lightgrey'
        matplotlib.rcParams['grid.linestyle'] = '-'
        
        matplotlib.rcParams['figure.subplot.left'] = 0.1
        matplotlib.rcParams['figure.subplot.bottom'] = 0.13
        matplotlib.rcParams['figure.subplot.right'] = 0.9
        matplotlib.rcParams['figure.subplot.top'] = 0.9

    def _get_estimator(self, window, price_data, clean=True, use_overlapping_adjustment_factor=True):
        """Selector for volatility estimator
        
        Parameters
        ----------
        window : int
            Rolling window for which to calculate the estimator
        clean : boolean
            Set to True to remove the NaNs at the beginning of the series
        
        Returns
        -------
        y : pandas.DataFrame
            Estimator series values
        """

        return getattr(models, self._estimator).get_estimator(
            price_data=price_data,
            window=window,
            clean=clean,
            use_overlapping_adjustment_factor=use_overlapping_adjustment_factor
        )
   
    def cones(self, windows=[30, 60, 90, 120], quantiles=[0.25, 0.75], use_overlapping_adjustment_factor=True):
        """Plots volatility cones
        
        Parameters
        ----------
        windows : [int, int, ...]
            List of rolling windows for which to calculate the estimator cones
        quantiles : [lower, upper]
            List of lower and upper quantiles for which to plot the cones
        """

        price_data = self._price_data

        if len(windows) < 2:
            raise ValueError(
                'Two or more window periods required')
        if len(quantiles) != 2:
            raise ValueError(
                'A two element list of quantiles is required, lower and upper')
        if quantiles[0] + quantiles[1] != 1.0:
            raise ValueError(
                'The sum of the quantiles must equal 1.0')
        if quantiles[0] > quantiles[1]:
            raise ValueError(
                'The lower quantiles (first element) must be less than the upper quantile (second element)')
        
        max_ = []
        min_ = []
        top_q = []
        median = []
        bottom_q = []
        realized = []
        data = []

        for window in windows:
            
            estimator = self._get_estimator(
                window=window,
                price_data=price_data,
                clean=True,
                use_overlapping_adjustment_factor=use_overlapping_adjustment_factor
            )

            max_.append(estimator.max())
            top_q.append(estimator.quantile(quantiles[1]))
            median.append(estimator.median())
            bottom_q.append(estimator.quantile(quantiles[0]))
            min_.append(estimator.min())
            realized.append(estimator[-1])

            data.append(estimator)
        
        if self._estimator == "Skew" or self._estimator == "Kurtosis":
            f = lambda x: "%i" % round(x, 0)
        else:
            f = lambda x: "%i%%" % round(x*100, 0)

        # figure
        fig = plt.figure(figsize=(8, 6))
        fig.autofmt_xdate()
        left, width = 0.07, 0.65
        bottom, height = 0.2, 0.7
        left_h = left+width+0.02
        rect_cones = [left, bottom, width, height]
        rect_box = [left_h, bottom, 0.17, height]
        cones = plt.axes(rect_cones)
        box = plt.axes(rect_box)

        # set the plots
        cones.plot(windows, max_, label="Max")
        cones.plot(windows, top_q, label=str(int(quantiles[1]*100)) + " Prctl")
        cones.plot(windows, median, label="Median")
        cones.plot(windows, bottom_q, label=str(int(quantiles[0]*100)) + " Prctl")
        cones.plot(windows, min_, label="Min")
        cones.plot(windows, realized, 'r-.', label="Realized")

        # set the x ticks and limits
        cones.set_xticks(windows)
        cones.set_xlim((windows[0]-5, windows[-1]+5))

        # set and format the y-axis labels
        locs = cones.get_yticks()
        cones.set_yticklabels(map(f, locs))

        # turn on the grid
        cones.grid(True, axis='y', which='major', alpha=0.5)

        # set the title
        cones.set_title(self._estimator + ' (' + self._symbol + ', daily ' + self._start + ' to ' + self._end + ')')

        # set the legend
        cones.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)

        # box plot
        box.boxplot(data, notch=1, sym='+')
        box.plot([i for i in range(1, len(windows)+1)], realized, color='r', marker='*', markeredgecolor='k')

        # set and format the y-axis labels
        locs = box.get_yticks()
        box.set_yticklabels(map(f, locs))

        # move the y-axis ticks on the right side
        box.yaxis.tick_right()

        # turn on the grid
        box.grid(True, axis='y', which='major', alpha=0.5)
        
        return fig, plt, data

    def rolling_quantiles(self, window=30, quantiles=[0.25, 0.75], use_overlapping_adjustment_factor=True):
        """Plots rolling quantiles of volatility
        
        Parameters
        ----------
        window : int
            Rolling window for which to calculate the estimator
        quantiles : [lower, upper]
            List of lower and upper quantiles for which to plot
        """

        price_data = self._price_data

        if len(quantiles) != 2:
            raise ValueError(
                'A two element list of quantiles is required, lower and upper')
        if quantiles[0] + quantiles[1] != 1.0:
            raise ValueError(
                'The sum of the quantiles must equal 1.0')
        if quantiles[0] > quantiles[1]:
            raise ValueError(
                'The lower quantiles (first element) must be less than the upper quantile (second element)')
        
        estimator = self._get_estimator(
            window=window,
            price_data=price_data,
            clean=True,
            use_overlapping_adjustment_factor=use_overlapping_adjustment_factor
        )
        date = estimator.index
        
        top_q = estimator.rolling(window=window, center=False).quantile(quantiles[1])
        median = estimator.rolling(window=window, center=False).median()
        bottom_q = estimator.rolling(window=window, center=False).quantile(quantiles[0])
        realized = estimator
        last = estimator[-1]

        if self._estimator == "Skew" or self._estimator == "Kurtosis":
            f = lambda x: "%i" % round(x, 0)
        else:
            f = lambda x: "%i%%" % round(x*100, 0)

        # figure
        fig = plt.figure(figsize=(8, 6))
        fig.autofmt_xdate()
        left, width = 0.07, 0.65
        bottom, height = 0.2, 0.7
        left_h = left+width+0.02
        
        rect_cones = [left, bottom, width, height]
        rect_box = [left_h, bottom, 0.17, height]
        
        cones = plt.axes(rect_cones)
        box = plt.axes(rect_box)

        # set the plots
        cones.plot(date, top_q, label=str(int(quantiles[1]*100)) + " Prctl")
        cones.plot(date, median, label="Median")
        cones.plot(date, bottom_q, label=str(int(quantiles[0]*100)) + " Prctl")
        cones.plot(date, realized, 'r-.', label="Realized")
        
        # set and format the y-axis labels
        locs = cones.get_yticks()
        cones.set_yticklabels(map(f, locs))
        
        # turn on the grid
        cones.grid(True, axis='y', which='major', alpha=0.5)
        
        # set the title
        cones.set_title(self._estimator + ' (' + self._symbol + ', daily ' + self._start + ' to ' + self._end + ')')
        
        # set the legend
        cones.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)

        # box plots
        box.boxplot(realized, notch=1, sym='+')
        box.plot(1, last, color='r', marker='*', markeredgecolor='k')
        
        # set and format the y-axis labels
        locs = box.get_yticks()
        box.set_yticklabels(map(f, locs))
        
        # move the y-axis ticks on the right side
        box.yaxis.tick_right()
        
        # turn on the grid
        box.grid(True, axis='y', which='major', alpha=0.5)
        
        return fig, plt

    def rolling_extremes(self, window=30, use_overlapping_adjustment_factor=True):
        """Plots rolling max and min of volatility estimator
        
        Parameters
        ----------
        window : int
            Rolling window for which to calculate the estimator
        """

        price_data = self._price_data

        estimator = self._get_estimator(
            window=window,
            price_data=price_data,
            clean=True,
            use_overlapping_adjustment_factor=use_overlapping_adjustment_factor
        )
        date = estimator.index
        max_ = estimator.rolling(window=window, center=False).max()
        min_ = estimator.rolling(window=window, center=False).min()
        realized = estimator
        last = estimator[-1]

        if self._estimator == "Skew" or self._estimator == "Kurtosis":
            f = lambda x: "%i" % round(x, 0)
        else:
            f = lambda x: "%i%%" % round(x*100, 0)

        # figure
        fig = plt.figure(figsize=(8, 6))
        fig.autofmt_xdate()
        left, width = 0.07, 0.65
        bottom, height = 0.2, 0.7
        left_h = left+width+0.02
        
        rect_cones = [left, bottom, width, height]
        rect_box = [left_h, bottom, 0.17, height]
        
        cones = plt.axes(rect_cones)
        box = plt.axes(rect_box)

        # set the plots
        cones.plot(date, max_, label="Max")
        cones.plot(date, min_, label="Min")
        cones.plot(date, realized, 'r-.', label="Realized")
        
        # set and format the y-axis labels
        locs = cones.get_yticks()
        cones.set_yticklabels(map(f, locs))
        
        # turn on the grid
        cones.grid(True, axis='y', which='major', alpha=0.5)
        
        # set the title
        cones.set_title(self._estimator + ' (' + self._symbol + ', daily ' + self._start + ' to ' + self._end + ')')
        
        # set the legend
        cones.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)

        # box plot
        box.boxplot(realized, notch=1, sym='+')
        box.plot(1, last, color='r', marker='*', markeredgecolor='k')
        
        # set and format the y-axis labels
        locs = box.get_yticks()
        box.set_yticklabels(map(f, locs))
        
        # move the y-axis ticks on the right side
        box.yaxis.tick_right()
        
        # turn on the grid
        box.grid(True, axis='y', which='major', alpha=0.5)
        
        return fig, plt

    def rolling_descriptives(self, window=30, use_overlapping_adjustment_factor=True):
        """Plots rolling first and second moment of volatility estimator
        
        Parameters
        ----------
        window : int
            Rolling window for which to calculate the estimator
        """

        price_data = self._price_data

        estimator = self._get_estimator(
            window=window,
            price_data=price_data,
            clean=True,
            use_overlapping_adjustment_factor=use_overlapping_adjustment_factor
        )
        date = estimator.index
        mean = estimator.rolling(window=window, center=False).mean()
        std = estimator.rolling(window=window, center=False).std()
        z_score = (estimator - mean) / std
        
        realized = estimator
        last = estimator[-1]

        if self._estimator == "Skew" or self._estimator == "Kurtosis":
            f = lambda x: "%i" % round(x, 0)
        else:
            f = lambda x: "%i%%" % round(x*100, 0)

        # figure
        fig = plt.figure(figsize=(8, 6))
        fig.autofmt_xdate()
        left, width = 0.07, 0.65
        left_h = left+width+0.02
        
        rect_cones = [left, 0.35, width, 0.55]
        rect_box = [left_h, 0.15, 0.17, 0.75]
        rect_z = [left, 0.15, width, 0.15]
        
        cones = plt.axes(rect_cones)
        box = plt.axes(rect_box)
        z = plt.axes(rect_z)
        
        if self._estimator == "Skew" or self._estimator == "Kurtosis":
            f = lambda x: "%i" % round(x, 0)
        else:
            f = lambda x: "%i%%" % round(x*100, 0)

        # set the plots
        cones.plot(date, mean, label="Mean")
        cones.plot(date, std, label="Std. Dev.")
        cones.plot(date, realized, 'r-.', label="Realized")
        
        # set and format the y-axis labels
        locs = cones.get_yticks()
        cones.set_yticklabels(map(f, locs))
        
        # turn on the grid
        cones.grid(True, axis='y', which='major', alpha=0.5)
        
        # set the title
        cones.set_title(self._estimator + ' (' + self._symbol + ', daily ' + self._start + ' to ' + self._end + ')')
        
        # shrink the plot up a bit and set the legend
        pos = cones.get_position()
        cones.set_position([pos.x0, pos.y0 + pos.height * 0.1, pos.width, pos.height * 0.9]) #
        cones.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)

        # box plot
        box.boxplot(realized, notch=1, sym='+')
        box.plot(1, last, color='r', marker='*', markeredgecolor='k')
        
        # set and format the y-axis labels
        locs = box.get_yticks()
        box.set_yticklabels(map(f, locs))
        
        # move the y-axis ticks on the right side
        box.yaxis.tick_right()
        
        # turn on the grid
        box.grid(True, axis='y', which='major', alpha=0.5)

        # z-score set the plots
        z.plot(date, z_score, 'm-', label="Z-Score")
        
        # turn on the grid
        z.grid(True, axis='y', which='major', alpha=0.5)
        
        # create a horizontal line at y=0
        z.axhline(0, 0, 1, linestyle='-', linewidth=1.0, color='black')
        
        # set the legend
        z.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3)
        
        return fig, plt

    def histogram(self, window=90, bins=100, normed=True, use_overlapping_adjustment_factor=True):
        """
        
        Parameters
        ----------
        window : int
            Rolling window for which to calculate the estimator
        bins : int
            
        """

        price_data = self._price_data

        estimator = self._get_estimator(
            window=window,
            price_data=price_data,
            clean=True,
            use_overlapping_adjustment_factor=use_overlapping_adjustment_factor
        )
        mean = estimator.mean()
        std = estimator.std()
        last = estimator[-1]

        fig = plt.figure(figsize=(8, 6))
        
        n, bins, patches = plt.hist(estimator, bins, facecolor='blue', alpha=0.25)
        
        if normed:
            y = norm.pdf(bins, mean, std)
            plt.plot(bins, y, 'g--', linewidth=1)

        plt.axvline(last, 0, 1, linestyle='-', linewidth=1.5, color='r')

        plt.grid(True, axis='y', which='major', alpha=0.5)
        plt.title('Distribution of ' + self._estimator +
                  ' estimator values (' + self._symbol +
                  ', daily ' + self._start + ' to ' + self._end + ')')
        
        return fig, plt
    
    def benchmark_compare(self, window=90, use_overlapping_adjustment_factor=True):
        """
        
        Parameters
        ----------
        window : int
            Rolling window for which to calculate the estimator
        bins : int
            
        """

        price_data = self._price_data
        bench_data = self._bench_data

        y = self._get_estimator(
            window=window,
            price_data=price_data,
            clean=True,
            use_overlapping_adjustment_factor=use_overlapping_adjustment_factor
        )
        x = self._get_estimator(
            window=window,
            price_data=bench_data,
            clean=True,
            use_overlapping_adjustment_factor=use_overlapping_adjustment_factor
        )
        date = y.index
        
        ratio = y / x

        if self._estimator == "Skew" or self._estimator == "Kurtosis":
            f = lambda x: "%i" % round(x, 0)
        else:
            f = lambda x: "%i%%" % round(x*100, 0)
        
        # figure
        fig = plt.figure(figsize=(8, 6))
        fig.autofmt_xdate()
        left, width = 0.07, .9
        
        rect_cones = [left, 0.4, width, .5]
        rect_box = [left, 0.15, width, 0.15]
        
        cones = plt.axes(rect_cones)
        box = plt.axes(rect_box)

        # set the plots
        cones.plot(date, y, label=self._symbol.upper())
        cones.plot(date, x, label=self._bench_symbol)
        
        # set and format the y-axis labels
        locs = cones.get_yticks()
        cones.set_yticklabels(map(f, locs))
        
        # turn on the grid
        cones.grid(True, axis='y', which='major', alpha=0.5)
        
        # set the title
        cones.set_title(self._estimator + ' (' + self._symbol +
                        ' v. ' + self._bench_symbol + ', daily ' +
                        self._start + ' to ' + self._end + ')')
        
        # shrink the plot up a bit and set the legend
        cones.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)

        # set the plot
        box.plot(date, ratio, label=self._symbol.upper() + '/' + self._bench_symbol)
        
        # set the y-limits
        box.set_ylim((ratio.min() - 0.05, ratio.max() + 0.05))
        
        # fill the area
        box.fill_between(date, ratio, 0, color='blue', alpha=0.25)
        
        # set the legend
        box.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3)

        return fig, plt

    def benchmark_correlation(self, window=90, use_overlapping_adjustment_factor=True):
        """
        
        Parameters
        ----------
        window : int
            Rolling window for which to calculate the estimator
        bins : int
            
        """
        
        price_data = self._price_data
        bench_data = self._bench_data

        y = self._get_estimator(
            window=window,
            price_data=price_data,
            clean=True,
            use_overlapping_adjustment_factor=use_overlapping_adjustment_factor
        )
        x = self._get_estimator(
            window=window,
            price_data=bench_data,
            clean=True,
            use_overlapping_adjustment_factor=use_overlapping_adjustment_factor
        )
        date = y.index

        corr = x.rolling(window=window).corr(other=y)

        if self._estimator == "Skew" or self._estimator == "Kurtosis":
            f = lambda x: "%i" % round(x, 0)
        else:
            f = lambda x: "%i%%" % round(x*100, 0)
        
        # figure
        fig = plt.figure(figsize=(8, 6))
        cones = plt.axes()

        # set the plots
        cones.plot(date, corr)

        # set the y-limits
        cones.set_ylim((corr.min() - 0.05, corr.max() + 0.05))

        # set and format the y-axis labels
        locs = cones.get_yticks()
        cones.set_yticklabels(map(f, locs))

        # turn on the grid
        cones.grid(True, axis='y', which='major', alpha=0.5)

        # set the title
        cones.set_title(self._estimator + ' (Correlation of ' +
                        self._symbol + ' v. ' + self._bench_symbol +
                        ', daily ' + self._start + ' to ' + self._end + ')')
        
        return fig, plt

    def benchmark_regression(self, window=90, use_overlapping_adjustment_factor=True):
        """
        
        Parameters
        ----------
        window : int
            Rolling window for which to calculate the estimator
        bins : int
            
        """
        price_data = self._price_data
        bench_data = self._bench_data

        y = self._get_estimator(
            window=window,
            price_data=price_data,
            clean=True,
            use_overlapping_adjustment_factor=use_overlapping_adjustment_factor
        )
        X = self._get_estimator(
            window=window,
            price_data=bench_data,
            clean=True,
            use_overlapping_adjustment_factor=use_overlapping_adjustment_factor
        )
        
        model = sm.OLS(y, X)
        results = model.fit()

        return results.summary()
    
    def term_sheet(
            self,
            window=30,
            windows=[30, 60, 90, 120],
            quantiles=[0.25, 0.75],
            bins=100,
            normed=True,
            use_overlapping_adjustment_factor=True):
        
        cones_fig, cones_plt, _ = self.cones(windows=windows, quantiles=quantiles, use_overlapping_adjustment_factor=use_overlapping_adjustment_factor)
        rolling_quantiles_fig, rolling_quantiles_plt = self.rolling_quantiles(window=window, quantiles=quantiles, use_overlapping_adjustment_factor=use_overlapping_adjustment_factor)
        rolling_extremes_fig, rolling_extremes_plt = self.rolling_extremes(window=window, use_overlapping_adjustment_factor=use_overlapping_adjustment_factor)
        rolling_descriptives_fig, rolling_descriptives_plt = self.rolling_descriptives(window=window, use_overlapping_adjustment_factor=use_overlapping_adjustment_factor)
        histogram_fig, histogram_plt = self.histogram(window=window, bins=bins, normed=normed, use_overlapping_adjustment_factor=use_overlapping_adjustment_factor)
        benchmark_compare_fig, benchmark_compare_plt = self.benchmark_compare(window=window, use_overlapping_adjustment_factor=use_overlapping_adjustment_factor)
        benchmark_corr_fig, benchmark_corr_plt = self.benchmark_correlation(window=window, use_overlapping_adjustment_factor=use_overlapping_adjustment_factor)
        benchmark_regression = self.benchmark_regression(window=window)
        
        filename = self._symbol.upper() + f'_{self._estimator}_termsheet_' + datetime.datetime.today().strftime("%Y%m%d") + '.pdf'
        fn = os.path.abspath(os.path.join(u'term-sheets', filename))
        pp = PdfPages(fn)
        
        pp.savefig(cones_fig)
        pp.savefig(rolling_quantiles_fig)
        pp.savefig(rolling_extremes_fig)
        pp.savefig(rolling_descriptives_fig)
        pp.savefig(histogram_fig)
        pp.savefig(benchmark_compare_fig)
        pp.savefig(benchmark_corr_fig)

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        ax.text(
            0, .2,
            benchmark_regression,
            family='monospace',
            fontsize=9
        )

        plt.axis('off')
        fig.tight_layout()
        pp.savefig(fig)
        pp.close()
        
        print('%s output complete' % filename)
