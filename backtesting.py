from src.pipeline import RiskPipeline
from src.risk_engine import RiskEngine
from src.models import RiskConfig, RiskReport

import pandas as pd
import numpy as np
from scipy import stats

class Backtesting:
    def __init__(self, risk_engine: RiskEngine, risk_report: RiskReport, window=250):
        self.risk_engine = risk_engine
        self.risk_report = risk_report
        self.window = window

    def rolling_backtest(self, asset_returns: pd.DataFrame) -> pd.DataFrame:
        returns = asset_returns.copy().dropna()
        weights = self.risk_report.weights

        returns = returns[weights.index]
        portfolio_returns = returns @ weights

        self.breach = pd.DataFrame({
                "actual_return": portfolio_returns,
                "var_return": pd.NA,
                "breach": pd.NA
            }, index=returns.index)

        for t in range(self.window, len(returns)):
            date = returns.index[t]

            estimation_window = returns.iloc[t - self.window : t]

            var_return = self.risk_engine.parametric_var(
                asset_returns=estimation_window,
                weights=weights,
            ).var_return

            actual_return = portfolio_returns.iloc[t]

            self.breach.at[date, "var_return"] = var_return
            self.breach.at[date, "breach"] = actual_return < var_return

        return self.breach.dropna()
            
    def binomial_test(self):
        x = self.breach['breach'].sum() 
        n = len(self.breach)
        p = 1 - self.risk_engine.confidence_level

        res = stats.binomtest(x, n=n, p=p, alternative='two-sided')
        p_value = res.pvalue
        
        return p_value

    def kupiec_test(self):
        n_exceptions = self.breach['breach'].sum() 
        n_observations = len(self.breach)
        p_expected = 1 - self.risk_engine.confidence_level
        p_empirical = n_exceptions / n_observations 

        # Formula: -2 * ln((p_expected^x * (1-p_expected)^(n-x)) / (p_empirical^x * (1-p_empirical)^(n-x)))
        term1 = n_exceptions * np.log(p_expected) + (n_observations - n_exceptions) * np.log(1 - p_expected)
        term2 = n_exceptions * np.log(p_empirical) + (n_observations - n_exceptions) * np.log(1 - p_empirical)
        lr_statistic = -2 * (term1 - term2)

        p_value = 1 - stats.chi2.cdf(lr_statistic, df=1)

        return lr_statistic, p_value
        

    def christoffersen_test(self):
        pass

    def traffic_light_test(self):
        pass


def main():
    from src.market_data import MarketData
    from src.portfolio import Portfolio

    config = RiskConfig(
        portfolio_path=r"data\raw\portfolio\portfolio.csv",
        start_date=None,
        end_date="2026-06-17",
        lookback_days=252,
        confidence_level=0.99,
        num_simulations=10000,
        random_seed=42,
        num_worst_days=5,
    )
    port = Portfolio(config.portfolio_path)

    
    md = MarketData(tickers=port.ticker_list,start_date="2021-06-17",end_date=config.end_date)

    returns = md.get_asset_returns()
    latest_prices = md.get_latest_prices()
    port.process_port(latest_prices)
    summary = port.portfolio_summary()

    pipeline = RiskPipeline(config)
    risk_report = pipeline.run()
    re = RiskEngine(portfolio_value=summary.net_exposure, confidence_level=config.confidence_level)
    backtesting = Backtesting(risk_engine=re, risk_report=risk_report)
    
    backtesting.rolling_backtest(returns)
    print(backtesting.binomial_test())


if __name__ == "__main__":
    main()
