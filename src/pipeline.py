from src.portfolio import Portfolio
from src.market_data import MarketData
from src.risk_engine import RiskEngine

from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd

@dataclass
class RiskReport:
    portfolio_summary: dict
    historical: dict
    parametric: dict
    monte_carlo: dict
    worst_days: pd.DataFrame

class RiskPipeline:
    def __init__(self, config):
        self.config = config

    def run(self):
        if self.config["start_date"] is None:
            start_date = (datetime.strptime(self.config["end_date"], "%Y-%m-%d")
                           - timedelta(days=self.config["lookback_days"])
                           ).strftime("%Y-%m-%d")
        else:
            start_date = self.config["start_date"]

        portfolio = Portfolio(self.config["portfolio_path"])
        market_data = MarketData(tickers=portfolio.ticker_list,
                                      start_date=start_date,
                                      end_date=self.config["end_date"])

        asset_returns = market_data.get_asset_returns()
        latest_prices = market_data.get_latest_prices()
        portfolio.process_port(latest_prices)
        portfolio_returns = portfolio.calculate_portfolio_returns(asset_returns)
        portfolio_summary = portfolio.portfolio_summary()
        portfolio_value = portfolio_summary['gross_exposure']
        weights = portfolio.get_weights(asset_returns.columns)
        
        risk_engine = RiskEngine(portfolio_returns=portfolio_returns,
                                      asset_returns=asset_returns,
                                      weights=weights,
                                      portfolio_value=portfolio_value,
                                      confidence_level=self.config['confidence_level'])

        historical_var = risk_engine.historical_var()
        parametric_var = risk_engine.parametric_var()
        monte_carlo_var = risk_engine.monte_carlo_var(n=self.config["num_simulations"], 
                                                        seed=self.config["random_seed"])
        worst_days = risk_engine.worst_days(n=self.config["num_worst_days"])

        return RiskReport(
            portfolio_summary=portfolio_summary,
            historical=historical_var,
            parametric=parametric_var,
            monte_carlo=monte_carlo_var,
            worst_days=worst_days
        )


def main():
    config = {
    "portfolio_path": r"data\raw\portfolio\portfolio.csv",
    "start_date": None,
    "end_date": "2026-06-17",
    "lookback_days": 252,
    "confidence_level": 0.99,
    "num_simulations": 10000,
    "random_seed": 42,
    "num_worst_days": 10
}
    rp = RiskPipeline(config)
    out = rp.run()
    print(out)

if __name__ == "__main__":
    main()