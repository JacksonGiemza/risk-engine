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

    def run(self, returns):
        historical_bt = self.rolling_backtest(returns, method="historical")
        parametric_bt = self.rolling_backtest(returns, method="parametric")
        monte_carlo_bt = self.rolling_backtest(returns, method="monte_carlo")

        return {
            "historical_var": {
                "binomial": self.binomial_test(historical_bt),
                "kupiec": self.kupiec_test(historical_bt),
                "christoffersen": self.christoffersen_test(historical_bt),
                "traffic_light": self.traffic_light_test(historical_bt)
            },
            "parametric_var": {
                "binomial": self.binomial_test(parametric_bt),
                "kupiec": self.kupiec_test(parametric_bt),
                "christoffersen": self.christoffersen_test(parametric_bt),
                "traffic_light": self.traffic_light_test(parametric_bt)
            },
            "monte_carlo_var": {
                "binomial": self.binomial_test(monte_carlo_bt),
                "kupiec": self.kupiec_test(monte_carlo_bt),
                "christoffersen": self.christoffersen_test(monte_carlo_bt),
                "traffic_light": self.traffic_light_test(monte_carlo_bt)
            },
        }

    def rolling_backtest(self, asset_returns: pd.DataFrame, method: str = "parametric") -> pd.DataFrame:
        returns = asset_returns.copy().dropna()
        weights = self.risk_report.weights

        returns = returns[weights.index]
        portfolio_returns = returns @ weights

        self.breach = pd.DataFrame(
            {
                "actual_return": portfolio_returns,
                "var_return": pd.NA,
                "breach": pd.NA,
            },
            index=returns.index,
        )

        for t in range(self.window, len(returns)):
            date = returns.index[t]

            asset_window = returns.iloc[t - self.window : t]
            portfolio_window = portfolio_returns.iloc[t - self.window : t]

            if method == "historical":
                var_return = self.risk_engine.historical_var(
                    portfolio_returns=portfolio_window,
                ).var_return

            elif method == "parametric":
                var_return = self.risk_engine.parametric_var(
                    asset_returns=asset_window,
                    weights=weights,
                ).var_return

            elif method == "monte_carlo":
                var_return = self.risk_engine.monte_carlo_var(
                    asset_returns=asset_window,
                    weights=weights,
                ).var_return

            else:
                raise ValueError(
                    "method must be one of: 'historical', 'parametric', 'monte_carlo'"
                )

            actual_return = portfolio_returns.iloc[t]

            self.breach.at[date, "var_return"] = var_return
            self.breach.at[date, "breach"] = actual_return < var_return

        return self.breach.dropna()
            
    def binomial_test(self, bt):
        x = bt['breach'].sum() 
        n = len(bt)
        p = 1 - self.risk_engine.confidence_level

        res = stats.binomtest(x, n=n, p=p, alternative='two-sided')
        p_value = res.pvalue
        
        return {"p_value": p_value}

    def kupiec_test(self, bt):
        n_exceptions = bt['breach'].sum() 
        n_observations = len(bt)
        p_expected = 1 - self.risk_engine.confidence_level
        p_empirical = n_exceptions / n_observations 

        # Formula: -2 * ln((p_expected^x * (1-p_expected)^(n-x)) / (p_empirical^x * (1-p_empirical)^(n-x)))
        term1 = n_exceptions * np.log(p_expected) + (n_observations - n_exceptions) * np.log(1 - p_expected)
        term2 = n_exceptions * np.log(p_empirical) + (n_observations - n_exceptions) * np.log(1 - p_empirical)
        lr_statistic = -2 * (term1 - term2)

        p_value = 1 - stats.chi2.cdf(lr_statistic, df=1)

        return {
            "lr_statistic": lr_statistic,
            "p_value": p_value
        }
    
    def christoffersen_test(self, bt):
        N = len(bt)
        N1 = bt['breach'].sum()
        N0 = N - N1
        pi_target = 1 - self.risk_engine.confidence_level

        pi_hat = N1 / N
        pi_hat = max(min(pi_hat, 1 - 1e-10), 1e-10) 
        
        ln_L_null_uc = N1 * np.log(pi_target) + N0 * np.log(1 - pi_target)
        ln_L_alt_uc = N1 * np.log(pi_hat) + N0 * np.log(1 - pi_hat)
        
        lr_uc = -2 * (ln_L_null_uc - ln_L_alt_uc)
        p_val_uc = 1 - stats.chi2.cdf(lr_uc, df=1)

        exceptions = bt["breach"].dropna().astype(int).to_numpy()

        y_tm1 = exceptions[:-1]
        y_t = exceptions[1:]

        n00 = np.sum((y_tm1 == 0) & (y_t == 0))
        n01 = np.sum((y_tm1 == 0) & (y_t == 1))
        n10 = np.sum((y_tm1 == 1) & (y_t == 0))
        n11 = np.sum((y_tm1 == 1) & (y_t == 1))
        print(n00, n01, n10, n11)
        pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
        pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
        pi_combined = (n01 + n11) / (n00 + n01 + n10 + n11)

        pi01 = max(min(pi01, 1 - 1e-10), 1e-10)
        pi11 = max(min(pi11, 1 - 1e-10), 1e-10)
        pi_combined = max(min(pi_combined, 1 - 1e-10), 1e-10)

        ln_L_null_ind = (n00 + n10) * np.log(1 - pi_combined) + (n01 + n11) * np.log(pi_combined)
        ln_L_alt_ind = (n00 * np.log(1 - pi01) + n01 * np.log(pi01) + 
                        n10 * np.log(1 - pi11) + n11 * np.log(pi11))

        lr_ind = -2 * (ln_L_null_ind - ln_L_alt_ind)
        p_val_ind = 1 - stats.chi2.cdf(lr_ind, df=1)

        lr_cc = lr_uc + lr_ind
        p_val_cc = 1 - stats.chi2.cdf(lr_cc, df=2)
        return {
            "Unconditional Coverage": {"LR": lr_uc, "p-value": p_val_uc},
            "Independence": {"LR": lr_ind, "p-value": p_val_ind},
            "Conditional Coverage": {"LR": lr_cc, "p-value": p_val_cc}
        }

    def traffic_light_test(self, bt):
        alpha = 1 - self.risk_engine.confidence_level
        
        N = len(bt)          
        X = int(bt['breach'].sum()) 
        
        cum_prob = stats.binom.cdf(X, N, alpha)
        
        if X <= 4:
            zone = "Green"
            multiplier_plus = 0.00
        elif cum_prob <= 0.9999:
            zone = "Yellow"
            yellow_multipliers = {5: 0.40, 6: 0.50, 7: 0.65, 8: 0.75, 9: 0.85}
            multiplier_plus = yellow_multipliers.get(X, 0.50) 
        else:
            zone = "Red"
            multiplier_plus = 1.00

        return {
            "Total Days (N)": N,
            "Observed Violations (X)": X,
            "Expected Violations": N * alpha,
            "Cumulative Probability": cum_prob,
            "Basel Zone": zone,
            "Capital Scaling Penalty (+/-)": multiplier_plus
        }


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
    print(backtesting.run(returns))

if __name__ == "__main__":
    main()
