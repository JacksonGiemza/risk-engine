from dataclasses import dataclass

@dataclass
class PricingResult:
    market_value: float
    abs_exposure: float

@dataclass
class Position:
    symbol: str
    asset_class: str
    quantity: int
    currency: str
    # side: str
    market_price: float
    instrument_type: str
    