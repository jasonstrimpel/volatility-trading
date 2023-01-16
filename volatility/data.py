import pandas


def yahoo_helper(symbol, data_path, *args):
    """
    Returns DataFrame/Panel of historical stock prices from symbols, over date
    range, start to end. 
    
    Parameters
        ----------
        symbol : string
            Single stock symbol (ticker)
        data_path: string
            Path to Yahoo! historical data CSV file
        *args:
            Additional arguments to pass to pandas.read_csv
    """

    try:

        # NOTE: one might be tempted to use Adj Close here... and that would be wrong.
        # Adj Close could be a value outside of the low to high range, causing these 
        # volitility estimators to make absolute nonsense. 

        data = pandas.read_csv(
            data_path,
            parse_dates=['Date'],
            index_col='Date',
            usecols=[
                'Date',
                'Open',
                'High',
                'Low',
                'Close',
                # 'Adj Close',
            ],
            *args
        )
        # .rename(columns={
        #     'Adj Close': 'Close'
        # })

    except Exception as e:
        raise e

    data.symbol = symbol
    return data
