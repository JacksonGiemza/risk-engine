import pandas as pd
import numpy as np
from scipy.stats import norm

class RiskEngine:
    def __init__(self, portfolio_returns, asset_returns, weights, portfolio_value, confidence_level):
        if portfolio_value > 0:
            self.portfolio_value = portfolio_value
        else:
            raise ValueError("portfolio_value should be greater than 0.")

        if 0 < confidence_level < 1:
            self.confidence_level = confidence_level
        else:
            raise ValueError("confidence_level should be between 0 and 1.")
        
        self.portfolio_returns = portfolio_returns

        if self.portfolio_returns.empty:
            raise ValueError("Portfolio Returns Series is Empty.")

        self.asset_returns = asset_returns.copy().dropna()

        if self.asset_returns.empty:
            raise ValueError("Asset Returns DataFrame is Empty.")

        self.tail_probability = 1 - self.confidence_level

        self.weights = weights

        self.cov_matrix = self.asset_returns.cov()
        self.cov_matrix.index.name = None
        self.cov_matrix.columns.name = None

        self.mean_vector = self.asset_returns.mean()


    def historical_var(self):
        var_return = float(self.portfolio_returns.quantile(self.tail_probability))
        var_dollars  = float(self.portfolio_value * abs(var_return))

        summary = {
            'method': 'Historical',
            'confidence_level': round(self.confidence_level, 2),
            'tail_probability': self.tail_probability,
            'var_return': round(var_return, 3),
            'var_percent': round(abs(var_return), 3),
            'var_dollars': round(var_dollars, 2)
        } 

        return summary

    # def expected_shortfall(self):
    #     if self.var_return is None or self.portfolio_returns.empty:
    #         raise ValueError("Run historical_var before expected_shortfall.")

    #     tail_losses = self.portfolio_returns[self.portfolio_returns <= self.var_return]
    #     expected_shortfall = float(tail_losses.mean())
    #     es_dollars = self.portfolio_value * abs(expected_shortfall)

    #     summary = {
    #        'method': 'Historical Expected Shortfall',
    #        'confidence_level': round(self.confidence_level, 2),
    #        'es_return': round(expected_shortfall, 3),
    #        'es_percent': round(abs(expected_shortfall), 3),
    #        'es_dollars': round(es_dollars, 2)
    #     }
    #     return summary
    
    def worst_days(self, n=10):
        worst_return = self.portfolio_returns[self.portfolio_returns < 0].nsmallest(n)
        worst_dollar = abs(worst_return * self.portfolio_value)

        worst_df = pd.DataFrame({'Return': worst_return, 'Dollar Loss': worst_dollar})

        return worst_df
    
    def parametric_var(self):

        portfolio_variance = float((self.weights @ self.cov_matrix @ self.weights.T).iloc[0, 0])
        portfolio_volatility = float(np.sqrt(portfolio_variance))

        z_score = float(norm.ppf(1 - self.tail_probability))

        var_percent = z_score * portfolio_volatility
        var_return = -var_percent
        var_dollars = float(z_score * portfolio_volatility * self.portfolio_value)
        
        summary = {
            'method': 'Parametric',
            'confidence_level': round(self.confidence_level, 2),
            'tail_probability': self.tail_probability,
            'portfolio_variance': portfolio_variance,
            'portfolio_volatility': portfolio_volatility,
            'z_score': z_score,
            'var_return': var_return,
            'var_percent': var_percent,
            'var_dollars': round(var_dollars, 2)
        } 

        return summary
    
    def monte_carlo_var(self, n=10000):
        np.random.seed(42)
        sim_returns = np.random.multivariate_normal(self.mean_vector, self.cov_matrix, n)
        sim_returns = pd.DataFrame(sim_returns, columns=self.cov_matrix.columns)

        sim_port_returns = sim_returns @ self.weights.T

        var_return = float(sim_port_returns.quantile(self.tail_probability).item())
        var_dollars = float(abs(var_return) * self.portfolio_value)

        summary = {
            'method': 'Monte Carlo',
            'confidence_level': round(self.confidence_level, 2),
            'tail_probability': self.tail_probability,
            'num_simulation': n,
            'var_return': var_return,
            'var_percent': -var_return,
            'var_dollars': round(var_dollars, 2)
        }
        
        return summary


def main():
    from market_data import MarketData
    from portfolio import Portfolio

    port = Portfolio(r"data\raw\portfolio\portfolio.csv")

    md = MarketData(tickers=port.ticker_list, start_date='2026-05-18',end_date='2026-06-17')
    returns = md.get_asset_returns()


    latest_prices = md.get_latest_prices()
    port.process_port(latest_prices)    

    port_returns = port.calculate_portfolio_returns(returns)
    port_value = port.portfolio_summary()['net_exposure']
    weights = port.get_weights(returns.columns)

    re = RiskEngine(portfolio_returns=port_returns, asset_returns=returns, weights=weights, portfolio_value=port_value, confidence_level=0.99)

    print(re.historical_var())
    print()
    print(re.parametric_var())
    print()
    print(re.monte_carlo_var())

if __name__ == "__main__":
    main()
