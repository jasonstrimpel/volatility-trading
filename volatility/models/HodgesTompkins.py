import math
import datetime

import pandas
import numpy as np

import data

def get_estimator(symbol, start, end, window=30, clean=True):
	
	prices = data.get_data(symbol, start, end)
	
	log_return = (prices['Close'] / prices['Close'].shift(1)).apply(np.log)

	vol = log_return.rolling(window=window, center=False).std() * math.sqrt(252)
	adj_factor = math.sqrt((1.0 / (1.0 - (window / (log_return.count() - (window - 1.0)))+(window**2 - 1.0)/(3.0 * (log_return.count() - (window - 1.0))**2))))

	result = vol * adj_factor
	result[:window-1] = np.nan
	
	if clean:
		return result.dropna()
	else:
		return result