from src.pipeline import RiskPipeline
from src.models import RiskConfig

class Backtesting:
    def __init__(self, RiskEngine, window=250):
        self.risk_engine = RiskEngine()
        self.window = window

    def create_breach_series(self, returns):
        
        for t in range(self.window, len(returns)):
            pass
        

    def basic_backtest_summary(self):
        pass

    def kupiec_test(self):
        pass

    def christoffersen_test(self):
        pass

    def traffic_light_test(self):
        pass


def main():
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
    pipeline = RiskPipeline(config)
    risk_report = pipeline.run()
    backtesting = Backtesting(risk_report.portfolio_returns, risk_report.historical)

    print(backtesting.create_breach_series())


if __name__ == "__main__":
    main()
