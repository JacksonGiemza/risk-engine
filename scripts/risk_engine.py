import pandas as pd
import numpy as np
from scipy.stats import norm


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
    
    def parametric_var(self, returns, portfolio):
        asset_returns = returns.copy().dropna()

        w_vector = portfolio[['symbol','weight']].T
        w_vector.columns = w_vector.iloc[0]
        w_vector = w_vector.drop(w_vector.index[0])
        w_vector = w_vector.reindex(columns=asset_returns.columns)

        cov_matrix = asset_returns.cov()
        cov_matrix.index.name = None
        cov_matrix.columns.name = None

        portfolio_variance = float((w_vector @ cov_matrix @ w_vector.T).iloc[0, 0])
        portfolio_volatility = float(np.sqrt(portfolio_variance))

        z_score = float(norm.ppf(1 - self.tail_probability))

        var_dollars = z_score * portfolio_volatility * self.portfolio_value
        
        summary = {
            'method': 'Parametric',
            'confidence_level': round(self.confidence_level, 2),
            'tail_probability': self.tail_probability,
            'portfolio_variance': portfolio_variance,
            'portfolio_volatility': portfolio_volatility,
            'z_score': z_score,
            'var_return': z_score * portfolio_volatility,
            'ver_percent': -(z_score * portfolio_volatility),
            'var_dollars': round(var_dollars, 2)
        } 

        return summary
    
def main():
    port = Portfolio(r"data\portfolio.csv")
    port.process_port()    

    md = MarketData(tickers=port.ticker_list, start_date='2026-05-18',end_date='2026-06-17')
    returns = md.get_asset_returns()

    port_summary = port.portfolio_summary()
    port_returns = port.calculate_portfolio_returns(returns)
    portfolio = port.portfolio

    re = RiskEngine(port_summary['net_exposure'], 0.99)
    print()
    print(re.historical_var(port_returns))
    print()
    print(re.parametric_var(returns, portfolio))

if __name__ == "__main__":
    main()
