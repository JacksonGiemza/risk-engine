from dataclasses import dataclass

@dataclass(slots=True)
class InstrumentMetadata:
    symbol: str
    instrument_type: str
    asset_class: str

@dataclass(slots=True)
class ETFMetadata(InstrumentMetaData):
    expense_ratio: float
    issuer: str
    
@dataclass(slots=True)
class FutureMetadata(InstrumentMetaData):
    multiplier: float
    exchange: str

@dataclass(slots=True)
class FXMetadata(InstrumentMetaData):
    base_currency: str
    quote_currency: str

@dataclass(slots=True)
class OptionMetadata(InstrumentMetaData):
    strike: float
    expiry: str
    multiplier: float
    option_type: str
    exercise_style: str
    