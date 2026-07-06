from dataclasses import dataclass

@dataclass(slots=True)
class InstrumentMetadata:
    symbol: str
    instrument_type: str
    asset_class: str

@dataclass(slots=True)
class ETFMetadata(InstrumentMetadata):
    expense_ratio: float
    issuer: str
    currency: str
    
@dataclass(slots=True)
class FutureMetadata(InstrumentMetadata):
    multiplier: float
    exchange: str
    currency: str

@dataclass(slots=True)
class FXMetadata(InstrumentMetadata):
    base_currency: str
    quote_currency: str

@dataclass(slots=True)
class OptionMetadata(InstrumentMetadata):
    strike: float
    expiry: str
    multiplier: float
    option_type: str
    exercise_style: str
    