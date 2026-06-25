import pandas as pd

class Portfolio:
    def __init__(self, PATH):
        self.PATH = PATH
        self.weighted_returns = pd.DataFrame()
        self.portfolio_returns = pd.DataFrame()
        self.portfolio, self.ticker_list = self._load_portfolio()

    # -- facade methods -- 
    def process_port(self, latest_prices):
        self._validate_portfolio(stage="loaded")
        self._attach_latest_prices(latest_prices)
        self._validate_portfolio(stage="priced")
        self._calculate_market_values()
        self._calculate_weights()
        self._validate_portfolio(stage="processed")
        return self.portfolio

    def calculate_portfolio_returns(self, asset_returns):
        self._validate_portfolio(stage="processed")
        asset_returns = asset_returns.copy().dropna(how="all")

        weights = self._get_weights(asset_returns.columns)    
        self.weighted_returns = asset_returns.mul(weights.iloc[0],axis=1)
        self.portfolio_returns = self.weighted_returns.sum(axis=1)

        return self.portfolio_returns

    def portfolio_summary(self):
        self._validate_portfolio(stage="processed")

        total_market_value = self.portfolio['market_value'].sum()
        long_exposure = self.portfolio.loc[self.portfolio['side'] == 'Long', 'market_value'].sum()
        short_exposure = self.portfolio.loc[self.portfolio['side'] == 'Short', 'market_value'].sum()
        gross_exposure = self.portfolio["abs_exposure"].sum()
        net_exposure = long_exposure + short_exposure

        summary = {
            "total_market_value": round(total_market_value, 2),
            "long_exposure": round(long_exposure, 2),
            "short_exposure": round(abs(short_exposure), 2),
            "gross_exposure": round(gross_exposure, 2),
            "net_exposure": round(net_exposure, 2),
            "net_exposure_ratio": round(net_exposure / gross_exposure, 2),
        }
        return summary

    # -- internal methods --
    def _load_portfolio(self):
        self.portfolio = pd.read_csv(self.PATH)

        self.portfolio['side'] = 'Long'
        self.portfolio.loc[self.portfolio['quantity'] < 0, 'side'] = 'Short'

        self.ticker_list = list(self.portfolio['symbol'].unique())

        return self.portfolio, self.ticker_list
    
    def _attach_latest_prices(self, latest_prices):
        self.portfolio['latest_price'] = None
        for symbol in self.ticker_list:
            try:
                current_price = latest_prices.loc[symbol]
                self.portfolio.loc[self.portfolio['symbol'] == symbol, 'latest_price'] = current_price
                
            except Exception as e:
                raise ValueError(f"No data for {symbol}: {e}, in latest prices.")

        return self.portfolio

    def _calculate_market_values(self):
        self.portfolio['market_value'] = self.portfolio['quantity'] * self.portfolio['latest_price']
        self.portfolio['abs_exposure'] = self.portfolio["market_value"].abs()
        self.gross_exposure = self.portfolio['abs_exposure'].sum()

        return self.portfolio

    def _calculate_weights(self):
        self.portfolio["weight"] = self.portfolio['market_value'] / self.gross_exposure
        return self.portfolio
    
    def _get_weights(self, returns_columns):
        weights = self.portfolio[['symbol','weight']].T
        weights.columns = weights.iloc[0]
        weights = weights.drop(weights.index[0]).reindex(columns=returns_columns)

        if weights.isna().any().any():
            raise ValueError("Missing weights for one or more return columns.")
        
        return weights
    
    def _validate_portfolio(self, stage="loaded"):
        if stage == "loaded":
            if self.portfolio.empty:
                raise ValueError("Portfolio is empty.")
            
            if 'symbol' not in self.portfolio.columns or 'quantity' not in self.portfolio.columns:
                raise ValueError("Portfolio missing required columns, symbol and or quantity.")
            
            if not self.portfolio['symbol'].is_unique:
                raise ValueError("Portfolio has duplicate symbols.")
            
            if not pd.api.types.is_numeric_dtype(self.portfolio['quantity']):
                raise ValueError("quantity is not numeric.")

        if stage == "priced":
            if 'latest_price' not in self.portfolio.columns:
                raise ValueError("latest_price column not found.")
            
            if self.portfolio['latest_price'].isna().any():
                raise ValueError("Missing values in portfolio latest_price.")
            
            if not pd.api.types.is_numeric_dtype(self.portfolio['latest_price']):
                raise ValueError("latest_price is not numeric.")

        if stage == "processed":
            if 'market_value' not in self.portfolio.columns:
                raise ValueError("market_value column not found.")
            
            if 'abs_exposure' not in self.portfolio.columns:
                raise ValueError("abs_exposure column not found.")
            
            if 'weight' not in self.portfolio.columns:
                raise ValueError("weight column not found.")
            
            if self.portfolio['abs_exposure'].sum() <= 0:
                raise ValueError("Portfolio gross exposure is 0 or less.")
            
            if self.portfolio['market_value'].isna().any():
                raise ValueError("Missing values in portfolio market_value.")
            
            if self.portfolio['abs_exposure'].isna().any():
                raise ValueError("Missing values in portfolio abs_exposure.")
            
            if self.portfolio['weight'].isna().any():
                raise ValueError("Missing values in portfolio weight.")
            
            


def main():
    from market_data import MarketData

    port = Portfolio(r"data\raw\portfolio\portfolio.csv")

    md = MarketData(tickers=port.ticker_list, start_date='2026-05-18',end_date='2026-06-17')
    returns = md.get_asset_returns()
    latest_prices = md.get_latest_prices()

    port.process_port(latest_prices)
    print(port.calculate_portfolio_returns(returns))


if __name__ == "__main__":
    main()
