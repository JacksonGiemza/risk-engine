from datetime import datetime, timedelta
import pandas as pd

from src.portfolio import Portfolio
from src.market_data import MarketData
from src.risk_engine import RiskEngine
from src.models import RiskReport, RiskConfig, PortfolioSummary, RiskMetrics


class RiskPipeline:
    def __init__(self, config: RiskConfig) -> None:
        self.config = config

    def run(self) -> RiskReport:
        start_date = self._resolve_start_date()

        portfolio = Portfolio(self.config.portfolio_path)

        market_data = MarketData(
            tickers=portfolio.ticker_list,
            start_date=start_date,
            end_date=self.config.end_date,
        )

        asset_returns = market_data.get_asset_returns()
        latest_prices = market_data.get_latest_prices()

        portfolio.process_port(latest_prices)

        portfolio_returns = portfolio.calculate_portfolio_returns(asset_returns)
        portfolio_summary = portfolio.portfolio_summary()
        portfolio_value = portfolio_summary.gross_exposure
        weights = portfolio.get_weights(asset_returns.columns)

        risk_engine = RiskEngine(
            portfolio_returns=portfolio_returns,
            asset_returns=asset_returns,
            weights=weights,
            portfolio_value=portfolio_value,
            confidence_level=self.config.confidence_level,
        )

        historical = risk_engine.historical_var()
        parametric = risk_engine.parametric_var()
        monte_carlo = risk_engine.monte_carlo_var(
            n=self.config.num_simulations,
            seed=self.config.random_seed)

        worst_days = risk_engine.worst_days(
            n=self.config.num_worst_days
        )

        risk_table = pd.DataFrame([
            {
                "Method": historical.method,
                "VaR Return": historical.var_return,
                "VaR $": historical.var_dollars,
                "ES Return": historical.es_return,
                "ES $": historical.es_dollars,
            },
            {
                "Method": parametric.method,
                "VaR Return": parametric.var_return,
                "VaR $": parametric.var_dollars,
                "ES Return": parametric.es_return,
                "ES $": parametric.es_dollars,
            },
            {
                "Method": monte_carlo.method,
                "VaR Return": monte_carlo.var_return,
                "VaR $": monte_carlo.var_dollars,
                "ES Return": monte_carlo.es_return,
                "ES $": monte_carlo.es_dollars,
            },
        ])

        return RiskReport(
            portfolio_summary=portfolio_summary,
            historical=historical,
            parametric=parametric,
            monte_carlo=monte_carlo,
            holdings=portfolio.portfolio.copy(),
            portfolio_returns=portfolio_returns.copy(),
            risk_table=risk_table,
            worst_days=worst_days,
        )

    def _resolve_start_date(self) -> str:
        if self.config.start_date is None:
            return (
                datetime.strptime(self.config.end_date, "%Y-%m-%d")
                - timedelta(days=self.config.lookback_days)
            ).strftime("%Y-%m-%d")

        return self.config.start_date
    