import pandas as pd
import yfinance as yf

class MarketData:
    def __init__(self, tickers, start_date, end_date):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.prices = pd.DataFrame()
        self.returns = pd.DataFrame()

    # -- facade methods --
    def get_returns(self):
        self._get_price_history()
        returns = self._calculate_returns()
        return returns
    
    # -- internal methods for data processing --
    def _get_price_history(self):
        # get data from yahoo finance api using initialized variables
        history = yf.download(self.tickers, start=self.start_date, end=self.end_date,auto_adjust=True)

        # populate prices df with the close data from history
        self.prices = history['Close']
        
        return self.prices

    # def _save_prices_to_csv(self):
    #     self.prices.to_csv(rf'../data/port_ticker_history_{self.start_date}_{self.end_date}.csv')

    def _validate_prices(self): 
        # check if prices df is empty, and raise value error if it is
        if self.prices.empty:
            raise ValueError("Prices DataFrame is empty.")
        
        # get shape tuple of prices df
        rows, columns = self.prices.shape
        
        # check the date index of prices for duplicates and return num of unique duplicates
        duplicate_dates = len(self.prices.index[self.prices.index.duplicated()].unique())
        
        if duplicate_dates > 0:
            raise ValueError("Duplicate dates within prices df")

        # find missing values
        missing_values = self.prices.isna().sum().sum()

        start, end = (
        self.prices.index.min(),
        self.prices.index.max()
        )
        
        info = {
            "rows": rows,
            "columns": columns,
            "missing_values": missing_values,
            "duplicate_dates": duplicate_dates,
            "is_sorted": self.prices.index.is_monotonic_increasing,
            "start_date": start,
            "end_date": end
        }

        return info

    def _calculate_returns(self):
        # validate data
        self._validate_prices()

        if self.prices.empty:
            raise ValueError("Prices DataFrame is empty.")
        
        self.returns = (self.prices / self.prices.shift(1)) - 1

        return self.returns
    
def main():
    tickers = ['EEM', 'GLD', 'HYG', 'IWM', 'QQQ', 'SPY', 'TLT', 'USO', 'UUP']
    market = MarketData(tickers=tickers, start_date='2026-05-18',end_date='2026-06-17')
    returns = market.get_returns()
    print(returns)
    
if __name__ == "__main__":
    main()
    
