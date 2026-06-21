import pandas as pd
import yfinance as yf
from market_data import MarketData

class Portfolio:

    def __init__(self, PATH):
        self.portfolio = pd.DataFrame()
        self.ticker_list = None
        self.PATH = PATH

    def load_portfolio(self):
        self.portfolio = pd.read_csv(self.PATH)

        self.portfolio = self.portfolio.drop(['asset_class','currency','strategy'],axis=1)

        self.portfolio['side'] = 'Long'
        self.portfolio.loc[self.portfolio['quantity'] < 0, 'side'] = 'Short'

        self.ticker_list = self.portfolio['symbol'].unique()

        return self.portfolio
    
    def attach_latest_prices(self):
        
        self.portfolio['latest_price'] = None

        tickers = yf.Tickers(" ".join(self.ticker_list))
        
        for symbol in self.ticker_list:
            try:
                current_price = tickers.tickers[symbol].info.get('regularMarketPrice')
                self.portfolio.loc[self.portfolio['symbol'] == symbol, 'latest_price'] = current_price
                
            except Exception as e:
                print(f"Could not fetch data for {symbol}: {e}")

        return self.portfolio

    def calculate_market_values(self):
        self.portfolio['market_value'] = self.portfolio['quantity'] * self.portfolio['latest_price']
        self.portfolio['abs_exposure'] = abs(self.portfolio['quantity'] * self.portfolio['latest_price'])
        return self.portfolio

    def calculate_weights(self):
        total_port_value = self.portfolio['market_value'].sum()
        self.portfolio["weight"] = round(self.portfolio['market_value'] / total_port_value, 2)
        return self.portfolio

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
    
    def tickers(self):
        return list(self.portfolio['symbol'].unique())

    def calculate_portfolio_returns(self, returns):
        pass


if __name__ == "__main__":
    from market_data import MarketData
 
    port = Portfolio(r"Z:\Projects\risk-engine\data\portfolio.csv")
    port.load_portfolio()
    port.attach_latest_prices()
    port.calculate_market_values()
    portfolio = port.calculate_weights()
    returns = MarketData(tickers=port.tickers(), start_date='2026-05-18',end_date='2026-06-17').get_price_history()
    print(port.calculate_portfolio_returns(returns=returns))

