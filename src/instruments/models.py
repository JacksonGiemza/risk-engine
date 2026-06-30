from dataclasses import dataclass

@dataclass(slots=True)
class InstrumentMetaData:
    symbol: str
    instrument_type: str
    asset_class: str
    
@dataclass(slots=True)
class FutureMetaData(InstrumentMetaData):
    multiplier: float
    exchange: str

@dataclass(slots=True)
class EtfMetaData(InstrumentMetaData):
    expense_ratio: float
    issuer: str

@dataclass(slots=True)
class FxMetaData(InstrumentMetaData):
    base_currency: str
    quote_currency: str

@dataclass(slots=True)
class OptionsMetaData(InstrumentMetaData):
    strike: int
    expiry: str
    multiplier: int
    option_type: str
    exercise_style: str
    