import numpy as np
import pandas as pd
from scipy.stats import norm

from src.models import RiskMetrics


class RiskEngine:
    def __init__(
        self,
        portfolio_returns: pd.Series,
        asset_returns: pd.DataFrame,
        weights: pd.Series,
        portfolio_value: float,
        confidence_level: float,
    ) -> None:
        if portfolio_value <= 0:
            raise ValueError("portfolio_value should be greater than 0.")

        if not (0 < confidence_level < 1):
            raise ValueError("confidence_level should be between 0 and 1.")

        if portfolio_returns.empty:
            raise ValueError("Portfolio returns Series is empty.")

        if weights.empty:
            raise ValueError("Weights Series is empty.")

        self.portfolio_value: float = portfolio_value
        self.confidence_level: float = confidence_level
        self.tail_probability: float = 1 - confidence_level

        self.portfolio_returns: pd.Series = portfolio_returns.copy().dropna()

        if self.portfolio_returns.empty:
            raise ValueError("Portfolio returns Series is empty after dropping missing values.")

        self.asset_returns: pd.DataFrame = asset_returns.copy().dropna()

        if self.asset_returns.empty:
            raise ValueError("Asset returns DataFrame is empty after dropping missing values.")

        self.weights: pd.Series = weights

        self.covariance_matrix: pd.DataFrame = self.asset_returns.cov()
        self.covariance_matrix.index.name = None
        self.covariance_matrix.columns.name = None

        if not self.weights.index.equals(self.covariance_matrix.columns):
            raise ValueError("weights and covariance matrix columns are not aligned.")

        self.mean_returns: pd.Series = self.asset_returns.mean()

    def historical_var(self) -> RiskMetrics:
        var_return: float = float(self.portfolio_returns.quantile(self.tail_probability))
        var_dollars: float = float(abs(var_return) * self.portfolio_value)

        tail_losses: pd.Series = self.portfolio_returns[
            self.portfolio_returns <= var_return
        ]

        es_return: float = float(tail_losses.mean())
        es_dollars: float = float(abs(es_return) * self.portfolio_value)

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

    def parametric_var(self) -> RiskMetrics:
        variance: float = float(
            self.weights.T @ self.covariance_matrix @ self.weights
        )

        volatility: float = float(np.sqrt(variance))
        z_score: float = float(norm.ppf(1 - self.tail_probability))

        var_percent: float = z_score * volatility
        var_return: float = -var_percent
        var_dollars: float = float(var_percent * self.portfolio_value)

        es_percent: float = float(
            (volatility * norm.pdf(z_score)) / self.tail_probability
        )
        es_return: float = -es_percent
        es_dollars: float = float(es_percent * self.portfolio_value)

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

    def monte_carlo_var(
        self,
        n: int = 10_000,
        seed: int = 42,
    ) -> RiskMetrics:
        np.random.seed(seed)

        simulated_asset_returns = np.random.multivariate_normal(
            self.mean_returns,
            self.covariance_matrix,
            n,
        )

        simulated_asset_returns = pd.DataFrame(
            simulated_asset_returns,
            columns=self.covariance_matrix.columns,
        )

        simulated_portfolio_returns: pd.Series = simulated_asset_returns @ self.weights

        var_return: float = float(
            simulated_portfolio_returns.quantile(self.tail_probability)
        )
        var_dollars: float = float(abs(var_return) * self.portfolio_value)

        tail_losses: pd.Series = simulated_portfolio_returns[
            simulated_portfolio_returns <= var_return
        ]

        es_return: float = float(tail_losses.mean())
        es_dollars: float = float(abs(es_return) * self.portfolio_value)

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

    def worst_days(self, n: int = 10) -> pd.DataFrame:
        worst_returns: pd.Series = self.portfolio_returns[
            self.portfolio_returns < 0
        ].nsmallest(n)

        worst_dollars: pd.Series = abs(worst_returns * self.portfolio_value)

        worst_df: pd.DataFrame = pd.DataFrame(
            {
                "Return": worst_returns,
                "Dollar Loss": worst_dollars,
            }
        )

        return worst_df