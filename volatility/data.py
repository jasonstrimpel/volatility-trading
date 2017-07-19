def get_data(symbol, start, end, adjust_price=True, interval='d'):
    """
    Returns DataFrame/Panel of historical stock prices from symbols, over date
    range, start to end. 
    
    Parameters
        ----------
        symbols : string, array-like object (list, tuple, Series), or DataFrame
            Single stock symbol (ticker), array-like object of symbols or
            DataFrame with index containing stock symbols.
        start : string, (defaults to '1/1/2010')
            Starting date, timestamp. Parses many different kind of date
            representations (e.g., 'JAN-01-2010', '1/1/10', 'Jan, 1, 1980')
        end : string, (defaults to today)
            Ending date, timestamp. Same format as starting date.
    
        adjust_price : bool, default False
            If True, adjusts all prices in hist_data ('Open', 'High', 'Low',
            'Close') based on 'Close' price. Adds 'Adj_Ratio' column and drops
            'Adj Close'.

        interval : string, default 'd'
            Time interval code, valid values are 'd' for daily, 'w' for weekly,
            'm' for monthly and 'v' for dividend.
    
    """
    
    return False
    # data = YahooDailyReader(symbols=symbol, start=start, end=end,
    #                         adjust_price=adjust_price,
    #                         interval=interval).read()
    # del data['Adj_Ratio']
    # data['Return'] = (data['Close'] / data['Close'].shift(1)) - 1
    #
    # return data
