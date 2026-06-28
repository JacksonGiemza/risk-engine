import numpy as np
import pandas as pd
from scipy.stats import norm

from src.models import RiskMetrics


class RiskEngine:
    def __init__(self, portfolio_value: float, confidence_level: float, n: int = 10_000, seed: int = 42) -> None:
        if portfolio_value <= 0:
            raise ValueError("portfolio_value should be greater than 0.")

        if not (0 < confidence_level < 1):
            raise ValueError("confidence_level should be between 0 and 1.")

        self.portfolio_value = portfolio_value
        self.confidence_level = confidence_level
        self.tail_probability = 1 - confidence_level
        
        self.n = n
        self.seed = seed

    def historical_var(self, portfolio_returns) -> RiskMetrics:
        """
        Estimate portfolio VaR with Empirical Quantile
        """
        var_return = float(portfolio_returns.quantile(self.tail_probability))
        var_dollars = float(abs(var_return) * self.portfolio_value)

        tail_losses = portfolio_returns[portfolio_returns <= var_return]

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

    def parametric_var(self, weights: pd.Series, asset_returns: pd.DataFrame) -> RiskMetrics:
        """
        Estimate portfolio VaR with Inverse Cumulative Distribution Function
        """
        asset_returns = asset_returns.copy().dropna()
        covariance_matrix = asset_returns.cov()
        covariance_matrix.index.name = None
        covariance_matrix.columns.name = None

        if not weights.index.equals(covariance_matrix.columns):
            raise ValueError("weights and covariance matrix columns are not aligned.")
        
        portfolio_mean = float(weights @ asset_returns.mean())

        variance = float(weights.T @ covariance_matrix @ weights)
        volatility = float(np.sqrt(variance))
        z_score = float(norm.ppf(1 - self.tail_probability))

        var_return = portfolio_mean - z_score * volatility
        var_percent = abs(var_return)
        var_dollars = float(var_percent * self.portfolio_value)

        es_return = float(portfolio_mean - (volatility * norm.pdf(z_score)) / self.tail_probability)
        es_percent = abs(es_return)
        es_dollars = float(es_percent * self.portfolio_value)

        return RiskMetrics(
            method="Parametric",
            confidence_level=self.confidence_level,
            tail_probability=self.tail_probability,
            var_return=var_return,
            var_percent=var_percent,
            var_dollars=var_dollars,
            es_return=es_return,
            es_percent=es_percent,
            es_dollars=es_dollars,
        )

    def monte_carlo_var(self, weights: pd.Series, asset_returns: pd.DataFrame) -> RiskMetrics:
        """
        Estimate VaR using multivariate Monte Carlo simulation
        """
        asset_returns = asset_returns.copy().dropna()
        covariance_matrix = asset_returns.cov()
        covariance_matrix.index.name = None
        covariance_matrix.columns.name = None
        
        if not weights.index.equals(covariance_matrix.columns):
            raise ValueError("weights and covariance matrix columns are not aligned.")
        
        mean_returns = asset_returns.mean()

        rng = np.random.default_rng(self.seed)
        simulated_asset_returns = rng.multivariate_normal(mean_returns, covariance_matrix, self.n)
        simulated_asset_returns = pd.DataFrame(simulated_asset_returns, columns=covariance_matrix.columns)
        simulated_portfolio_returns: pd.Series = simulated_asset_returns @ weights

        var_return = float(simulated_portfolio_returns.quantile(self.tail_probability))
        var_dollars = float(abs(var_return) * self.portfolio_value)

        tail_losses = simulated_portfolio_returns[
            simulated_portfolio_returns <= var_return
        ]

        es_return = float(tail_losses.mean())
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

    def worst_days(self, portfolio_returns, n: int = 10) -> pd.DataFrame:
        worst_returns = portfolio_returns[portfolio_returns < 0].nsmallest(n)
        worst_dollars = abs(worst_returns * self.portfolio_value)
        worst_df = pd.DataFrame(
            {
                "Return": worst_returns,
                "Dollar Loss": worst_dollars,
            }
        )
        return worst_df
    