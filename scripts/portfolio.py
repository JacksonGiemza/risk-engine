import pandas as pd

class Portfolio:
    def __init__(self, PATH):
        self.PATH = PATH
        self.weighted_returns = pd.DataFrame()
        self.portfolio_returns = pd.DataFrame()
        self.portfolio, self.ticker_list = self._load_portfolio()
    
    # -- facade methods -- 
    def process_port(self, latest_prices):
        self._load_portfolio()
        self._attach_latest_prices(latest_prices)
        self._calculate_market_values()
        self._calculate_weights()
        return self.portfolio

    def calculate_portfolio_returns(self, asset_returns):
        asset_returns = asset_returns.copy().dropna(how="all")

        weights = self.portfolio[['symbol','weight']].T
        weights.columns = weights.iloc[0]
        weights = weights.drop(weights.index[0]).reindex(columns=asset_returns.columns)

        self.weighted_returns = asset_returns.mul(weights.iloc[0],axis=1)
        self.portfolio_returns = self.weighted_returns.sum(axis=1)

        return self.portfolio_returns

    def portfolio_summary(self):
        total_market_value = self.portfolio['market_value'].sum()
        long_exposure = self.portfolio.loc[self.portfolio['side'] == 'Long', 'market_value'].sum()
        short_exposure = self.portfolio.loc[self.portfolio['side'] == 'Short', 'market_value'].sum()
        gross_exposure = abs(long_exposure) + abs(short_exposure)
        net_exposure = long_exposure + short_exposure

        summary = {
            "total_market_value": round(total_market_value, 2),
            "long_exposure": round(long_exposure, 2),
            "short_exposure": round(abs(short_exposure), 2),
            "gross_exposure": round(gross_exposure, 2),
            "net_exposure": round(net_exposure, 2),
            "gross_leverage": round(gross_exposure / total_market_value, 2),
            "net_leverage": round(net_exposure / total_market_value, 2)
        }
        return summary

    # -- internal methods --
    def _load_portfolio(self):
        self.portfolio = pd.read_csv(self.PATH)

        self.portfolio = self.portfolio.drop(['asset_class','currency','strategy'],axis=1)

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
                print(f"No data for {symbol}: {e}, in latest prices.")

        return self.portfolio

    def _calculate_market_values(self):
        self.portfolio['market_value'] = self.portfolio['quantity'] * self.portfolio['latest_price']
        self.portfolio['abs_exposure'] = abs(self.portfolio['quantity'] * self.portfolio['latest_price'])
        return self.portfolio

    def _calculate_weights(self):
        total_port_value = self.portfolio['market_value'].sum()
        self.portfolio["weight"] = self.portfolio['market_value'] / total_port_value
        return self.portfolio
    

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
