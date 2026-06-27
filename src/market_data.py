import pandas as pd
import yfinance as yf
from pathlib import Path
import hashlib

class MarketData:
    def __init__(self, tickers: list[str], start_date: str, end_date: str):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.prices = pd.DataFrame()
        self.returns = pd.DataFrame()
        self.raw_prices_file_path: Path | None = None
        self._create_dynamic_paths()

    # -- facade methods --
    def get_asset_returns(self, use_cache=True) -> pd.DataFrame:
        if use_cache and self._cache_exists():
            self._load_prices_from_cache()
        else:
            self._get_price_history()

        returns = self._calculate_returns()
        return returns
    
    def get_latest_prices(self) -> dict[str, float]:
        if self.prices.empty:
            raise ValueError("Load prices before getting latest prices.")
        
        return self.prices.iloc[-1].to_dict()
    
    # -- internal methods for data processing --
    def _get_price_history(self) -> pd.DataFrame:
        history = yf.download(self.tickers, start=self.start_date, end=self.end_date,auto_adjust=True)
        self.prices = history['Close']
        self._save_prices_to_cache()
        return self.prices
        
    def _validate_prices(self) -> dict[str, any]: 
        if self.prices.empty:
            raise ValueError("Prices DataFrame is empty.")
        
        rows, columns = self.prices.shape
        duplicate_dates = len(self.prices.index[self.prices.index.duplicated()].unique())
        
        if duplicate_dates > 0:
            raise ValueError("Duplicate dates within prices df")

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

    def _calculate_returns(self) -> pd.DataFrame:
        self._validate_prices()

        if self.prices.empty:
            raise ValueError("Prices DataFrame is empty.")
        
        self.returns = (self.prices / self.prices.shift(1)) - 1

        return self.returns
    
    ### caching methods for raw price data ###
    def _load_prices_from_cache(self) -> pd.DataFrame:
        if self._cache_exists():
            self.prices = pd.read_csv(self.raw_prices_file_path, index_col=0, parse_dates=True)
            return self.prices
        else: 
            raise ValueError(f"Cache for {self.raw_prices_file_path} does not exist.")
        
    def _save_prices_to_cache(self) -> None:
        self.raw_prices_file_path.parent.mkdir(parents=True, exist_ok=True)
        self.prices.to_csv(self.raw_prices_file_path)
    
    def _cache_exists(self) -> bool:
        return self.raw_prices_file_path is not None and self.raw_prices_file_path.is_file()

    def _create_dynamic_paths(self) -> None:
        root = Path(__file__).resolve().parents[1]
        data_path = root / "data"

        sorted_tickers = sorted([t.upper() for t in self.tickers])
        ticker_hash = hashlib.md5(",".join(sorted_tickers).encode()).hexdigest()[:8]
        self.raw_prices_file_path = Path(data_path / "raw" / "prices" / f"raw_prices_{ticker_hash}_{self.start_date}_to_{self.end_date}.csv")
    