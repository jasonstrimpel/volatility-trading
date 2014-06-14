import pandas
import pandas.io.data as pd

def get_data(ticker, start, end):

    data = pd.DataReader(ticker, "yahoo", start, end)

    adj_factor = data['Adj Close'] / data['Close']

    adj_open = data['Open'] * adj_factor
    adj_high = data['High'] * adj_factor
    adj_low = data['Low'] * adj_factor
    
    returns = (data['Adj Close'] / data['Adj Close'].shift(1)) - 1

    return pandas.DataFrame({
        'Open': data['Open'],
        'High': data['High'], 
        'Low': data['Low'],
        'Close': data['Close'],
        'Volume': data['Volume'],
        'Adj Open': adj_open,
        'Adj High': adj_high,
        'Adj Low': adj_low,
        'Adj Close': data['Adj Close'],
        'Return': returns
    }, index=data.index)
