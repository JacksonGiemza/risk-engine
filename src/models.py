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

    weights: pd.Series
    asset_returns: pd.DataFrame

@dataclass
class PortfolioSummary:
    total_market_value: float
    gross_exposure: float
    net_exposure: float
    long_exposure: float
    short_exposure: float
    net_exposure_ratio: float

@dataclass(frozen=True)
class RiskConfig:
    portfolio_path: str
    start_date: str | None
    end_date: str
    lookback_days: int
    confidence_level: float
    num_simulations: int
    random_seed: int
    num_worst_days: int  

@dataclass
class BinomialResult:
    p_value: float

@dataclass
class KupiecResult:
    lr_statistic: float
    p_value: float

@dataclass
class ChristoffersenComponent:
    lr: float
    p_value: float

@dataclass
class ChristoffersenResult:
    unconditional: ChristoffersenComponent
    independence: ChristoffersenComponent
    conditional: ChristoffersenComponent

@dataclass
class TrafficLightResult:
    total_days: int
    observed_violations: int
    expected_violations: float
    cumulative_probability: float
    zone: str
    capital_scaling_penalty: float

@dataclass
class BacktestResult:
    method: str

    binomial: BinomialResult
    kupiec: KupiecResult
    christoffersen: ChristoffersenResult
    traffic_light: TrafficLightResult