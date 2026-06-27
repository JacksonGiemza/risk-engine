import pandas as pd
import numpy as np
from scipy.stats import norm
from src.models import RiskMetrics


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

        if weights.empty:
            raise ValueError("Weights series is empty.")

        self.cov_matrix = self.asset_returns.cov()
        self.cov_matrix.index.name = None
        self.cov_matrix.columns.name = None

        if (self.weights.index != self.cov_matrix.columns).all():
            raise ValueError("weights and cov_matrix columns unaligned.")

        self.mean_vector = self.asset_returns.mean()


    def historical_var(self):
        var_return = float(self.portfolio_returns.quantile(self.tail_probability))
        var_dollars  = float(self.portfolio_value * abs(var_return))

        tail_losses = self.portfolio_returns[self.portfolio_returns <= var_return]
        es_return = float(tail_losses.mean())
        es_dollars = float(abs(es_return) * self.portfolio_value)

        return RiskMetrics(
            method="Historical",
            confidence_level=self.confidence_level,
            tail_probability=self.tail_probability,

            var_return=var_return,
            var_percent=abs(var_return),
            var_dollars=var_dollars,

            es_return=es_return,
            es_percent=abs(es_return),
            es_dollars=es_dollars,
        )
    
    def parametric_var(self):
        variance = self.weights.T @ self.cov_matrix @ self.weights
        volatility = float(np.sqrt(variance))

        z_score = float(norm.ppf(1 - self.tail_probability))

        var_percent = z_score * volatility
        var_return = -var_percent
        var_dollars = float(z_score * volatility * self.portfolio_value)
        
        es_percent = float((volatility * norm.pdf(z_score)) / self.tail_probability)
        es_dollars = float(es_percent * self.portfolio_value)

        return RiskMetrics(
            method="Parametric",
            confidence_level=self.confidence_level,
            tail_probability=self.tail_probability,

            var_return=var_return,
            var_percent=var_percent,
            var_dollars=var_dollars,

            es_return=-es_percent,
            es_percent=es_percent,
            es_dollars=es_dollars,
        )    
    
    def monte_carlo_var(self, n=10000, seed=42):
        np.random.seed(seed)
        sim = np.random.multivariate_normal(self.mean_vector, self.cov_matrix, n)
        sim = pd.DataFrame(sim, columns=self.cov_matrix.columns)

        sim_returns = sim @ self.weights

        var_return = float(sim_returns.quantile(self.tail_probability).item())
        var_dollars = float(abs(var_return) * self.portfolio_value)
        
        tail_loss = sim_returns[sim_returns <= var_return]
        es_return = float(tail_loss.mean())
        es_dollars = float(abs(es_return) * self.portfolio_value)

        return RiskMetrics(
            method="Monte Carlo",
            confidence_level=self.confidence_level,
            tail_probability=self.tail_probability,

            var_return=var_return,
            var_percent=abs(var_return),
            var_dollars=var_dollars,

            es_return=es_return,
            es_percent=abs(es_return),
            es_dollars=es_dollars,
        )    
    
    def worst_days(self, n=10):
        worst_return = self.portfolio_returns[self.portfolio_returns < 0].nsmallest(n)
        worst_dollar = abs(worst_return * self.portfolio_value)

        worst_df = pd.DataFrame({'Return': worst_return, 'Dollar Loss': worst_dollar})

        return worst_df


def main():
    from market_data import MarketData
    from portfolio import Portfolio

    port = Portfolio(r"data\raw\portfolio\portfolio.csv")

    md = MarketData(tickers=port.ticker_list, start_date='2026-05-18',end_date='2026-06-17')
    returns = md.get_asset_returns()


    latest_prices = md.get_latest_prices()
    port.process_port(latest_prices)    

    port_returns = port.calculate_portfolio_returns(returns)
    port_value = port.portfolio_summary()['gross_exposure']
    weights = port.get_weights(returns.columns)

    re = RiskEngine(portfolio_returns=port_returns, asset_returns=returns, weights=weights, portfolio_value=port_value, confidence_level=0.99)

    print(re.historical_var())
    print()
    print(re.parametric_var())
    print()
    print(re.monte_carlo_var())

if __name__ == "__main__":
    main()
