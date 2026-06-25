import pandas as pd
import yfinance as yf
from pathlib import Path
import hashlib

class MarketData:
    def __init__(self, tickers, start_date, end_date):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.prices = pd.DataFrame()
        self.returns = pd.DataFrame()
        self.raw_prices_file_path = None
        self._create_dynamic_paths()

    # -- facade methods --
    def get_asset_returns(self, use_cache=True):
        if use_cache and self._cache_exists():
            self._load_prices_from_cache()
        else:
            self._get_price_history()

        returns = self._calculate_returns()
        return returns
    
    def get_latest_prices(self):
        if self.prices.empty:
            raise ValueError("Load prices before getting latest prices.")
        
        return self.prices.iloc[-1]
    
    # -- internal methods for data processing --
    def _get_price_history(self):
        # get data from yahoo finance api using initialized variables
        history = yf.download(self.tickers, start=self.start_date, end=self.end_date,auto_adjust=True)

        # populate prices df with the close data from history
        self.prices = history['Close']
        
        self._save_prices_to_cache()
        return self.prices
        
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

        if not self.prices.index.is_monotonic_increasing:
            raise ValueError("Price index is not sorted oldest to newest.")
        
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
    
    ### caching methods for raw price data ###
    def _load_prices_from_cache(self):
        if self._cache_exists():
            self.prices = pd.read_csv(self.raw_prices_file_path, index_col=0, parse_dates=True)
            return self.prices
        else: 
            raise ValueError(f"Cache for {self.raw_prices_file_path} does not exist.")
        
    def _save_prices_to_cache(self):
        self.raw_prices_file_path.parent.mkdir(parents=True, exist_ok=True)
        self.prices.to_csv(self.raw_prices_file_path)
    
    def _cache_exists(self):
        if self.raw_prices_file_path.is_file():
            return True
        return False

    def _create_dynamic_paths(self):
        root = Path(__file__).resolve().parents[1]
        data_path = root / "data"

        # generate file name for raw prices
        sorted_tickers = sorted([t.upper() for t in self.tickers])
        ticker_hash = hashlib.md5(",".join(sorted_tickers).encode()).hexdigest()[:8]
        self.raw_prices_file_path = Path(data_path / "raw" / "prices" / f"raw_prices_{ticker_hash}_{self.start_date}_to_{self.end_date}.csv")
    
def main():
    tickers = ['EEM', 'GLD', 'HYG', 'IWM', 'QQQ', 'SPY', 'TLT', 'USO', 'UUP']
    market = MarketData(tickers=tickers, start_date='2026-05-18',end_date='2026-06-17')
    returns = market.get_asset_returns()
    print(returns)
    
if __name__ == "__main__":
    main()
    