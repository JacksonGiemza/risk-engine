from datetime import datetime, timedelta
import pandas as pd

from src.portfolio import Portfolio
from src.market_data import MarketData
from src.risk_engine import RiskEngine
from src.backtesting import Backtesting
from src.models import RiskReport, RiskConfig, BacktestReport


class RiskPipeline:
    def __init__(self, config: RiskConfig) -> None:
        self.config = config
        self.portfolio = Portfolio(self.config.portfolio_path)

    def run(self) -> RiskReport:
        if self.config.start_date is None:
            start_date = self._get_start_date(self.config.lookback_days)
        else:
            start_date = self.config.start_date

        market_data = MarketData(
            tickers=self.portfolio.ticker_list,
            start_date=start_date,
            end_date=self.config.end_date,
        )

        asset_returns = market_data.get_asset_returns()
        latest_prices = market_data.get_latest_prices()

        self.portfolio.process_port(latest_prices)

        portfolio_returns = self.portfolio.calculate_portfolio_returns(asset_returns)
        portfolio_summary = self.portfolio.portfolio_summary()
        portfolio_value = portfolio_summary.gross_exposure
        self.weights = self.portfolio.get_weights(asset_returns.columns)

        self.risk_engine = RiskEngine(
            portfolio_value=portfolio_value,
            confidence_level=self.config.confidence_level,
            n=self.config.num_simulations,
            seed=self.config.random_seed
        )

        historical = self.risk_engine.historical_var(portfolio_returns)
        parametric = self.risk_engine.parametric_var(weights=self.weights, asset_returns=asset_returns)
        monte_carlo = self.risk_engine.monte_carlo_var(weights=self.weights, asset_returns=asset_returns)

        worst_days = self.risk_engine.worst_days(
            portfolio_returns=portfolio_returns,
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
        self.risk_report = RiskReport(
            portfolio_summary=portfolio_summary,
            historical=historical,
            parametric=parametric,
            monte_carlo=monte_carlo,
            holdings=self.portfolio.portfolio.copy(),
            portfolio_returns=portfolio_returns.copy(),
            risk_table=risk_table,
            worst_days=worst_days,
            weights=self.weights,
            asset_returns=asset_returns
        )

        return self.risk_report
    
    def run_backtest(self) -> BacktestReport:
        if not hasattr(self, "risk_engine") or not hasattr(self, "risk_report"):
            self.run()

        start_date = self._get_start_date(self.config.backtest_lookback)
        market_data = MarketData(
            tickers=self.portfolio.ticker_list,
            start_date=start_date,
            end_date=self.config.end_date,
        )

        asset_returns = market_data.get_asset_returns()

        backtesting = Backtesting(
            risk_engine=self.risk_engine, 
            weights=self.weights, 
            window=self.config.lookback_days)
        
        backtest_report = backtesting.run(asset_returns=asset_returns)

        return backtest_report

    def _get_start_date(self, lookback) -> str:
        return (
            datetime.strptime(self.config.end_date, "%Y-%m-%d")
            - timedelta(days=lookback)
        ).strftime("%Y-%m-%d")

    