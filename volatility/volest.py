import datetime
import os
from StringIO import StringIO

import pandas

import models
import data
import numpy as np

import matplotlib
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.mathtext as text
from matplotlib.ticker import FuncFormatter
from matplotlib.font_manager import FontProperties
from matplotlib.backends.backend_pdf import PdfPages

# references
# http://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
# http://www.blog.pythonlibrary.org/2010/09/04/python-101-how-to-open-a-file-or-program/

class VolatilityEstimator(object):

    def __init__(self, ticker, start, end, type):
        '''Constructor for volatility estimators
        
        Parameters
        ----------
        ticker : string
            Stock ticker symbol valid on finance.yahoo.com
        start : datetime
            Start date for data collection
        end : datetime
            End date for data collection
        type : string
            Estimator type; valid arguments are:
                "GarmanKlass", "HodgesTompkins", "Kurtosis", "Parkinson", "Raw",
                "RogersSatchell", "Skew", "YangZhang"
        '''
        if ticker is None or ticker == '':
            raise ValueError('Ticker symbol required')
        if start is None or start == '':
            raise ValueError('Start date required')
        if end is None or end == '':
            raise ValueError('End date required')
        if type not in ["GarmanKlass", "HodgesTompkins", "Kurtosis", "Parkinson", "Raw", "RogersSatchell", "Skew", "YangZhang"]:
            raise ValueError('Acceptable volatility model is required')
        
        self._ticker = ticker
        self._start = start
        self._end = end
        self._type = type
        
        matplotlib.rc('image', origin='upper')
        
        matplotlib.rcParams['font.family'] = 'Times New Roman'
        matplotlib.rcParams['font.size'] = '11'
        
        matplotlib.rcParams['grid.color'] = 'lightgrey'
        matplotlib.rcParams['grid.linestyle'] = '-'
        
        matplotlib.rcParams['figure.subplot.left'] = 0.1
        matplotlib.rcParams['figure.subplot.bottom'] = 0.13
        matplotlib.rcParams['figure.subplot.right'] = 0.9
        matplotlib.rcParams['figure.subplot.top'] = 0.9

    def _get_estimator(self, window, ticker=None, clean=True):
        '''Selector for volatility estimator
        
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
        '''
        
        if not ticker:
            ticker = self._ticker
        
        if self._type is "GarmanKlass":
            return models.GarmanKlass.get_estimator(ticker=ticker, start=self._start, end=self._end, window=window, clean=clean)
        elif self._type is "HodgesTompkins":
            return models.HodgesTompkins.get_estimator(ticker=ticker, start=self._start, end=self._end, window=window, clean=clean)
        elif self._type is "Kurtosis":
            return models.Kurtosis.get_estimator(ticker=ticker, start=self._start, end=self._end, window=window, clean=clean)
        elif self._type is "Parkinson":
            return models.Parkinson.get_estimator(ticker=ticker, start=self._start, end=self._end, window=window, clean=clean)
        elif self._type is "Raw":
            return models.Raw.get_estimator(ticker=ticker, start=self._start, end=self._end, window=window, clean=clean)
        elif self._type is "RogersSatchell":
            return models.RogersSatchell.get_estimator(ticker=ticker, start=self._start, end=self._end, window=window, clean=clean)
        elif self._type is "Skew":
            return models.Skew.get_estimator(ticker=ticker, start=self._start, end=self._end, window=window, clean=clean)
        elif self._type is "YangZhang":
            return models.YangZhang.get_estimator(ticker=ticker, start=self._start, end=self._end, window=window, clean=clean)
   
    def cones(self, windows=[30, 60, 90, 120], quantiles=[0.25, 0.75]):
        '''Plots volatility cones
        
        Parameters
        ----------
        windows : [int, int, ...]
            List of rolling windows for which to calculate the estimator cones
        quantiles : [lower, upper]
            List of lower and upper quantiles for which to plot the cones
        '''
        if len(windows) < 2:
            raise ValueError('Two or more window periods required')
        if len(quantiles) != 2:
            raise ValueError('A two element list of quantiles is required, lower and upper')
        if quantiles[0] + quantiles[1] != 1.0:
            raise ValueError('The sum of the quantiles must equal 1.0')
        if quantiles[0] > quantiles[1]:
            raise ValueError('The lower quantiles (first element) must be less than the upper quantile (second element)')
        
        max = []
        min = []
        top_q = []
        median = []
        bottom_q = []
        realized = []
        data = []

        for w in windows:
            
            estimator = self._get_estimator(w)

            max.append(estimator.max())
            top_q.append(estimator.quantile(quantiles[1]))
            median.append(estimator.median())
            bottom_q.append(estimator.quantile(quantiles[0]))
            min.append(estimator.min())
            realized.append(estimator[-1])

            data.append(estimator)
        
        if self._type is "Skew" or self._type is "Kurtosis":
            f = lambda x: "%i" % round(x, 0)
        else:
            f = lambda x: "%i%%" % round(x*100, 0)
        
        '''
        Figure args
        '''
        
        fig = plt.figure(figsize=(8, 6))
        
        left, width = 0.07, 0.65
        bottom, height = 0.2, 0.7
        bottom_h = left_h = left+width+0.02
        
        rect_cones = [left, bottom, width, height]
        rect_box = [left_h, bottom, 0.17, height]
        
        cones = plt.axes(rect_cones)
        box = plt.axes(rect_box)
        
        '''
        Cones plot args
        '''
        # set the plots
        cones.plot(windows, max, label="Max")
        cones.plot(windows, top_q, label=str(int(quantiles[1]*100)) + " Prctl")
        cones.plot(windows, median, label="Median")
        cones.plot(windows, bottom_q, label=str(int(quantiles[0]*100)) + " Prctl")
        cones.plot(windows, min, label="Min")
        cones.plot(windows, realized, 'r-.', label="Realized")
        
        # set the x ticks and limits
        cones.set_xticks((windows))
        cones.set_xlim((windows[0]-5, windows[-1]+5))
        
        # set and format the y-axis labels
        locs = cones.get_yticks()
        cones.set_yticklabels(map(f, locs))
        
        # turn on the grid
        cones.grid(True, axis='y', which='major', alpha=0.5)
        
        # set the title
        cones.set_title(self._type + ' (' + self._ticker + ', daily ' + self._start.strftime("%Y-%m-%d") + ' to ' + self._end.strftime("%Y-%m-%d") +  ')')
        
        # set the legend
        pos = cones.get_position() #
        cones.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
        
        '''
        Box plot args
        '''
        
        # set the plots
        box.boxplot(data, notch=1, sym='+')
        box.plot([i for i in range(1, len(windows)+1)], realized, color='r', marker='*', markeredgecolor='k')
        
        # set and format the y-axis labels
        locs = box.get_yticks()
        box.set_yticklabels(map(f, locs))
        
        # move the y-axis ticks on the right side
        box.yaxis.tick_right()
        
        # turn on the grid
        box.grid(True, axis='y', which='major', alpha=0.5)
        
        return fig, plt

    def rolling_quantiles(self, window=30, quantiles=[0.25, 0.75]):
        '''Plots rolling quantiles of volatility
        
        Parameters
        ----------
        window : int
            Rolling window for which to calculate the estimator
        quantiles : [lower, upper]
            List of lower and upper quantiles for which to plot
        '''
        if len(quantiles) != 2:
            raise ValueError('A two element list of quantiles is required, lower and upper')
        if quantiles[0] + quantiles[1] != 1.0:
            raise ValueError('The sum of the quantiles must equal 1.0')
        if quantiles[0] > quantiles[1]:
            raise ValueError('The lower quantiles (first element) must be less than the upper quantile (second element)')
        
        estimator = self._get_estimator(window)
        date = estimator.index
        top_q = pandas.rolling_quantile(estimator, window, quantiles[1])
        median = pandas.rolling_median(estimator, window)
        bottom_q = pandas.rolling_quantile(estimator, window, quantiles[0])
        realized = estimator
        last = estimator[-1]

        if self._type is "Skew" or self._type is "Kurtosis":
            f = lambda x: "%i" % round(x, 0)
        else:
            f = lambda x: "%i%%" % round(x*100, 0)

        '''
        Figure args
        '''
        
        fig = plt.figure(figsize=(8, 6))
        
        left, width = 0.07, 0.65
        bottom, height = 0.2, 0.7
        bottom_h = left_h = left+width+0.02
        
        rect_cones = [left, bottom, width, height]
        rect_box = [left_h, bottom, 0.17, height]
        
        cones = plt.axes(rect_cones)
        box = plt.axes(rect_box)
        
        '''
        Cones plot args
        '''
        
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
        cones.set_title(self._type + ' (' + self._ticker + ', daily ' + self._start.strftime("%Y-%m-%d") + ' to ' + self._end.strftime("%Y-%m-%d") +  ')')
        
        # set the legend
        cones.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
        
        '''
        Box plot args
        '''
        
        # set the plots
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

    def rolling_extremes(self, window=30):
        '''Plots rolling max and min of volatility estimator
        
        Parameters
        ----------
        window : int
            Rolling window for which to calculate the estimator
        '''
        estimator = self._get_estimator(window)
        date = estimator.index
        max = pandas.rolling_max(estimator, window)
        min = pandas.rolling_min(estimator, window)
        realized = estimator
        last = estimator[-1]

        if self._type is "Skew" or self._type is "Kurtosis":
            f = lambda x: "%i" % round(x, 0)
        else:
            f = lambda x: "%i%%" % round(x*100, 0)

        '''
        Figure args
        '''
        
        fig = plt.figure(figsize=(8, 6))
        
        left, width = 0.07, 0.65
        bottom, height = 0.2, 0.7
        bottom_h = left_h = left+width+0.02
        
        rect_cones = [left, bottom, width, height]
        rect_box = [left_h, bottom, 0.17, height]
        
        cones = plt.axes(rect_cones)
        box = plt.axes(rect_box)
        
        '''
        Cones plot args
        '''
        
        # set the plots
        cones.plot(date, max, label="Max")
        cones.plot(date, min, label="Min")
        cones.plot(date, realized, 'r-.', label="Realized")
        
        # set and format the y-axis labels
        locs = cones.get_yticks()
        cones.set_yticklabels(map(f, locs))
        
        # turn on the grid
        cones.grid(True, axis='y', which='major', alpha=0.5)
        
        # set the title
        cones.set_title(self._type + ' (' + self._ticker + ', daily ' + self._start.strftime("%Y-%m-%d") + ' to ' + self._end.strftime("%Y-%m-%d") +  ')')
        
        # set the legend
        cones.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
        
        '''
        Box plot args
        '''
        
        # set the plots
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

    def rolling_descriptives(self, window=30):
        '''Plots rolling first and second moment of volatility estimator
        
        Parameters
        ----------
        window : int
            Rolling window for which to calculate the estimator
        '''
        estimator = self._get_estimator(window)
        date = estimator.index
        mean = pandas.rolling_mean(estimator, window)
        std = pandas.rolling_std(estimator, window)
        z_score = (estimator - mean) / std
        
        realized = estimator
        last = estimator[-1]

        if self._type is "Skew" or self._type is "Kurtosis":
            f = lambda x: "%i" % round(x, 0)
        else:
            f = lambda x: "%i%%" % round(x*100, 0)

        '''
        Figure args
        '''
        
        fig = plt.figure(figsize=(8, 6))
        
        left, width = 0.07, 0.65
        bottom, height = 0.1, .8
        bottom_h = left_h = left+width+0.02
        
        rect_cones = [left, 0.35, width, 0.55]
        rect_box = [left_h, 0.15, 0.17, 0.75]
        rect_z = [left, 0.15, width, 0.15]
        
        cones = plt.axes(rect_cones)
        box = plt.axes(rect_box)
        z = plt.axes(rect_z)
        
        if self._type is "Skew" or self._type is "Kurtosis":
            f = lambda x: "%i" % round(x, 0)
        else:
            f = lambda x: "%i%%" % round(x*100, 0)
        
        '''
        Cones plot args
        '''
        
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
        cones.set_title(self._type + ' (' + self._ticker + ', daily ' + self._start.strftime("%Y-%m-%d") + ' to ' + self._end.strftime("%Y-%m-%d") +  ')')
        
        # shrink the plot up a bit and set the legend
        pos = cones.get_position() #
        cones.set_position([pos.x0, pos.y0 + pos.height * 0.1, pos.width, pos.height * 0.9]) #
        cones.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
        
        '''
        Box plot args
        '''
        
        # set the plots
        box.boxplot(realized, notch=1, sym='+')
        box.plot(1, last, color='r', marker='*', markeredgecolor='k')
        
        # set and format the y-axis labels
        locs = box.get_yticks()
        box.set_yticklabels(map(f, locs))
        
        # move the y-axis ticks on the right side
        box.yaxis.tick_right()
        
        # turn on the grid
        box.grid(True, axis='y', which='major', alpha=0.5)
        
        '''
        Z-Score plot args
        '''
        
        # set the plots
        z.plot(date, z_score, 'm-', label="Z-Score")
        
        # turn on the grid
        z.grid(True, axis='y', which='major', alpha=0.5)
        
        # create a horizontal line at y=0
        z.axhline(0, 0, 1, linestyle='-', linewidth=1.0, color='black')
        
        # set the legend
        z.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3)
        
        return fig, plt

    def histogram(self, window=90, bins=100, normed=True):
        '''
        
        Parameters
        ----------
        window : int
            Rolling window for which to calculate the estimator
        bins : int
            
        '''
        estimator = self._get_estimator(window)
        mean = estimator.mean()
        std = estimator.std()
        last = estimator[-1]

        fig = plt.figure(figsize=(8, 6))
        
        n, bins, patches = plt.hist(estimator, bins, normed=normed, facecolor='blue', alpha=0.25)
        
        if normed:
            y = mlab.normpdf(bins, mean, std)
            l = plt.plot(bins, y, 'g--', linewidth=1)

        plt.axvline(last, 0, 1, linestyle='-', linewidth=1.5, color='r')

        plt.grid(True, axis='y', which='major', alpha=0.5)
        plt.title('Distribution of ' + self._type + ' estimator values (' + self._ticker + ', daily ' + self._start.strftime("%Y-%m-%d") + ' to ' + self._end.strftime("%Y-%m-%d") +  ')')
        
        return fig, plt
    
    def benchmark_compare(self, window=90, bench='^GSPC'):
        '''
        
        Parameters
        ----------
        window : int
            Rolling window for which to calculate the estimator
        bins : int
            
        '''
        
        y = self._get_estimator(window)
        x = self._get_estimator(window, ticker=bench)
        date = y.index
        
        ratio = y / x

        if self._type is "Skew" or self._type is "Kurtosis":
            f = lambda x: "%i" % round(x, 0)
        else:
            f = lambda x: "%i%%" % round(x*100, 0)
        
        '''
        Figure args
        '''
        
        fig = plt.figure(figsize=(8, 6))
        
        left, width = 0.07, .9
        
        rect_cones = [left, 0.4, width, .5]
        rect_box = [left, 0.15, width, 0.15]
        
        cones = plt.axes(rect_cones)
        box = plt.axes(rect_box)
        
        '''
        Cones plot args
        '''
        
        # set the plots
        cones.plot(date, y, label=self._ticker.upper())
        cones.plot(date, x, label=bench.upper())
        
        # set and format the y-axis labels
        locs = cones.get_yticks()
        cones.set_yticklabels(map(f, locs))
        
        # turn on the grid
        cones.grid(True, axis='y', which='major', alpha=0.5)
        
        # set the title
        cones.set_title(self._type + ' (' + self._ticker + ' v. ' + bench.upper() + ', daily ' + self._start.strftime("%Y-%m-%d") + ' to ' + self._end.strftime("%Y-%m-%d") +  ')')
        
        # shrink the plot up a bit and set the legend
        cones.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)

        '''
        Cones plot args
        '''
        
        # set the plot
        box.plot(date, ratio, label=self._ticker.upper() + '/' + bench.upper())
        
        # set the y-limits
        box.set_ylim((ratio.min() - 0.05, ratio.max() + 0.05))
        
        # fill the area
        box.fill_between(date, ratio, 0, color='blue', alpha=0.25)
        
        # set the legend
        box.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3)

        return fig, plt

    def benchmark_correlation(self, window=90, bench='^GSPC'):
        '''
        
        Parameters
        ----------
        window : int
            Rolling window for which to calculate the estimator
        bins : int
            
        '''
        
        y = self._get_estimator(window)
        x = self._get_estimator(window, ticker=bench)
        date = y.index

        corr = pandas.rolling_corr(x, y, window)

        if self._type is "Skew" or self._type is "Kurtosis":
            f = lambda x: "%i" % round(x, 0)
        else:
            f = lambda x: "%i%%" % round(x*100, 0)
        
        '''
        Figure args
        '''
        
        fig = plt.figure(figsize=(8, 6))
        cones = plt.axes()
        
        '''
        Cones plot args
        '''
        
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
        cones.set_title(self._type + ' (Correlation of ' + self._ticker + ' v. ' + bench.upper() + ', daily ' + self._start.strftime("%Y-%m-%d") + ' to ' + self._end.strftime("%Y-%m-%d") +  ')')
        
        return fig, plt

    def benchmark_regression(self, window=90, bench='^GSPC'):
        '''
        
        Parameters
        ----------
        window : int
            Rolling window for which to calculate the estimator
        bins : int
            
        '''
        y = self._get_estimator(window)
        x = self._get_estimator(window, ticker=bench)
        
        model = str(pandas.ols(y=y, x=x))

        return model
    
    def term_sheet(self, window=30, windows=[30, 60, 90, 120], quantiles=[0.25, 0.75], bins=100, normed=True, bench='^GSPC', open=False):
        
        cones_fig, cones_plt = self.cones(windows=windows, quantiles=quantiles)
        rolling_quantiles_fig, rolling_quantiles_plt = self.rolling_quantiles(window=window, quantiles=quantiles)
        rolling_extremes_fig, rolling_extremes_plt = self.rolling_extremes(window=window)
        rolling_descriptives_fig, rolling_descriptives_plt = self.rolling_descriptives(window=window)
        histogram_fig, histogram_plt = self.histogram(window=window, bins=bins, normed=normed)
        benchmark_compare_fig, benchmark_compare_plt = self.benchmark_compare(window=window, bench=bench)
        benchmark_corr_fig, benchmark_corr_plt = self.benchmark_correlation(window=window, bench=bench)
        benchmark_regression = self.benchmark_regression(window=window, bench=bench)

        filename = self._ticker.upper() + '_termsheet_' + datetime.datetime.today().strftime("%Y%m%d") + '.pdf'
        pp = PdfPages(filename)
        
        pp.savefig(cones_fig)
        pp.savefig(rolling_quantiles_fig)
        pp.savefig(rolling_extremes_fig)
        pp.savefig(rolling_descriptives_fig)
        pp.savefig(histogram_fig)
        pp.savefig(benchmark_compare_fig)
        pp.savefig(benchmark_corr_fig)
        #pp.savefig(benchmark_regression)
        
        pp.close()
        
        print filename + ' output complete'
