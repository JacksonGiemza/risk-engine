from dataclasses import dataclass
import pandas as pd


@dataclass
class RiskMetrics:
    method: str
    confidence_level: float
    tail_probability: float

    var_return: float
    var_percent: float
    var_dollars: float

    es_return: float
    es_percent: float
    es_dollars: float

@dataclass
class RiskReport:
    portfolio_summary: dict

    historical: RiskMetrics
    parametric: RiskMetrics
    monte_carlo: RiskMetrics

    holdings: pd.DataFrame
    portfolio_returns: pd.Series
    risk_table: pd.DataFrame
    worst_days: pd.DataFrame

@dataclass
class PortfolioSummary:
    total_market_value: float
    gross_exposure: float
    net_exposure: float
    long_exposure: float
    short_exposure: float
    net_exposure_ratio: float