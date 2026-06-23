import pandas as pd

from market_data import MarketData
from portfolio import Portfolio

class RiskEngine:
    def __init__(self, portfolio_value, confidence_level):
        self.portfolio_value = portfolio_value
        self.confidence_level = confidence_level

    def historical_var(self, portfolio_returns):
        tail_probability = 1 - self.confidence_level
        var_return = portfolio_returns.quantile(tail_probability)
        var_dollars  = self.portfolio_value * abs(var_return)

        summary = {
            'method': 'Historical',
            'confidence_level': round(self.confidence_level, 2),
            'tail_probability': tail_probability,
            'var_return': round(var_return.item(), 3),
            'var_percent': round(abs(var_return.item()), 3),
            'var_dollars': round(var_dollars.item(), 2)
        } 

        return summary
    
def main():
    port = Portfolio(r"data\portfolio.csv")
    port.process_port()    

    md = MarketData(tickers=port.ticker_list, start_date='2026-05-18',end_date='2026-06-17')
    returns = md.get_asset_returns()
    port_returns = port.calculate_portfolio_returns(returns)
    re = RiskEngine(100000, 0.99)

    print(re.historical_var(port_returns))

if __name__ == "__main__":
    main()
