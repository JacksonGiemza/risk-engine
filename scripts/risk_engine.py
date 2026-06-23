import pandas as pd

from market_data import MarketData
from portfolio import Portfolio

class RiskEngine:
    def __init__(self, portfolio_value, confidence_level):
        if portfolio_value > 0:
            self.portfolio_value = portfolio_value
        else:
            raise ValueError("portfolio_value should be greater than 0.")

        if 0 < confidence_level < 1:
            self.confidence_level = confidence_level
        else:
            raise ValueError("confidence_level should be between 0 and 1.")
        
        self.portfolio_returns = pd.Series()
        self.var_return = None
        self.tail_probability = 1 - self.confidence_level
        self.var_dollars = None


    def historical_var(self, returns):
        self.portfolio_returns = returns.copy()

        if self.portfolio_returns.empty:
            raise ValueError("Portfolio Returns Series is Empty.")

        self.var_return = float(self.portfolio_returns.quantile(self.tail_probability))
        self.var_dollars  = self.portfolio_value * abs(self.var_return)

        summary = {
            'method': 'Historical',
            'confidence_level': round(self.confidence_level, 2),
            'tail_probability': self.tail_probability,
            'var_return': round(self.var_return, 3),
            'var_percent': round(abs(self.var_return), 3),
            'var_dollars': round(self.var_dollars, 2)
        } 

        return summary

    def expected_shortfall(self):
        if self.var_return is None or self.portfolio_returns.empty:
            raise ValueError("Run historical_var before expected_shortfall.")

        tail_losses = self.portfolio_returns[self.portfolio_returns <= self.var_return]
        expected_shortfall = float(tail_losses.mean())
        es_dollars = self.portfolio_value * abs(expected_shortfall)

        summary = {
           'method': 'Historical Expected Shortfall',
           'confidence_level': round(self.confidence_level, 2),
           'es_return': round(expected_shortfall, 3),
           'es_percent': round(abs(expected_shortfall), 3),
           'es_dollars': round(es_dollars, 2)
        }
        return summary
    
    def worst_days(self, n=10):
        if self.portfolio_returns.empty:
            raise ValueError("Run historical_var before worst_days.")

        worst_return = self.portfolio_returns[self.portfolio_returns < 0].nsmallest(n)
        worst_dollar = abs(worst_return * self.portfolio_value)

        worst_df = pd.DataFrame({'Return': worst_return, 'Dollar Loss': worst_dollar})

        return worst_df


    
def main():
    port = Portfolio(r"data\portfolio.csv")
    port.process_port()    

    md = MarketData(tickers=port.ticker_list, start_date='2026-05-18',end_date='2026-06-17')
    returns = md.get_asset_returns()
    port_returns = port.calculate_portfolio_returns(returns)

    port_summary = port.portfolio_summary()

    re = RiskEngine(port_summary['net_exposure'], 0.99)

    hist_var = re.historical_var(port_returns)
    print(hist_var)

    es = re.expected_shortfall()
    print(re.worst_days())

if __name__ == "__main__":
    main()
