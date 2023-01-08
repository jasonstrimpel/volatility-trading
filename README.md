# volest #

## A complete set of volatility estimators based on Euan Sinclair's Volatility Trading. ##
http://www.amazon.com/gp/product/0470181990/tag=quantfinancea-20

The original version incorporated network data acquisition from Yahoo!Finance
from `pandas_datareader`. Yahoo! changed their API and broke `pandas_datareader`.

The changes allow you to specify your own data so you're not tied into equity
data from Yahoo! finance. If you're still using equity data, just download
a CSV from finance.yahoo.com and use the `data.yahoo_data_helper` method
to form the data properly.

### Volatility estimators include: ###

* Garman Klass
* Parkinson
* Rogers Satchell
* Yang Zhang
* Standard Deviation

Also includes

* Skew
* Kurtosis
* Correlation

For each of the estimators, plot:

* Probability cones
* Rolling quantiles
* Rolling extremes
* Rolling descriptive statistics
* Histogram
* Comparison against arbirary comparable
* Correlation against arbirary comparable
* Regression against arbirary comparable

Create a term sheet with all the metrics printed to a PDF.

### Page 1 - Volatility cones ###
![Capture-1](docs/img/1.png)

### Page 2 - Volatility rolling percentiles ###
![Capture-2](docs/img/2.png)

### Page 3 - Volatility rolling min and max ###
![Capture-3](docs/img/3.png)

### Page 4 - Volatility rolling mean, standard deviation and zscore ###
![Capture-4](docs/img/4.png)

### Page 5 - Volatility distribution ###
![Capture-5](docs/img/5.png)

### Page 6 - Volatility, benchmark volatility and ratio###
![Capture-6](docs/img/6.png)

### Page 7 - Volatility rolling correlation with benchmark ###
![Capture-7](docs/img/7.png)

### Page 3 - Volatility OLS results ###
![Capture-8](docs/img/8.png)

Example usage:

```python

from volatility import volest
import yfinance as yf

# data
symbol = 'JPM'
bench = 'SPY'
estimator = 'GarmanKlass'

# estimator windows
window = 30
windows = [30, 60, 90, 120]
quantiles = [0.25, 0.75]
bins = 100
normed = True

# use the yahoo helper to correctly format data from finance.yahoo.com
jpm_price_data = yf.Ticker(symbol).history(period="5y")
jpm_price_data.symbol = symbol
spx_price_data = yf.Ticker(bench).history(period="5y")
spx_price_data.symbol = bench

# initialize class
vol = volest.VolatilityEstimator(
    price_data=jpm_price_data,
    estimator=estimator,
    bench_data=spx_price_data
)

# call plt.show() on any of the below...
_, plt = vol.cones(windows=windows, quantiles=quantiles)
_, plt = vol.rolling_quantiles(window=window, quantiles=quantiles)
_, plt = vol.rolling_extremes(window=window)
_, plt = vol.rolling_descriptives(window=window)
_, plt = vol.histogram(window=window, bins=bins, normed=normed)

_, plt = vol.benchmark_compare(window=window)
_, plt = vol.benchmark_correlation(window=window)

# ... or create a pdf term sheet with all metrics in term-sheets/
vol.term_sheet(
    window,
    windows,
    quantiles,
    bins,
    normed
)

```

Hit me on twitter with comments, questions, issues @jasonstrimpel
