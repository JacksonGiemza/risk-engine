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
        return self.portfolio

    def calculate_weights(self):
        total_port_value = self.portfolio['market_value'].sum()
        
        self.portfolio["weight"] = round(self.portfolio['market_value'] / total_port_value, 2)

        return self.portfolio

    def calculate_exposure(self):
        marketdata = MarketData(tickers = self.ticker_list, start_date='2026-05-18',end_date='2026-06-17')
        returns = marketdata.get_price_history()

        

    def calculate_portfolio_returns(self):
        pass


if __name__ == "__main__":
    port = Portfolio(r"Z:\Projects\risk-engine\data\portfolio.csv")
    port.load_portfolio()

    port.attach_latest_prices()

    port.calculate_market_values()

    df = port.calculate_weights()
    print(df.head())
