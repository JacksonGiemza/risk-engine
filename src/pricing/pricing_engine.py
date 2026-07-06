from src.pricing.linear_pricers import LinearPricer
from src.pricing.currency_conversion import CurrencyConverter
from src.instruments.instrument_loader import InstrumentLoader
from src.pricing.models import Position, PricingResult

class PricingEngine:
    def __init__(self, instrument_loader: InstrumentLoader, currency_converter: CurrencyConverter):
        self.instrument_loader = instrument_loader
        self.currency_converter = currency_converter

    def price_position(self, position: Position) -> PricingResult:
        
        metadata = self.instrument_loader.load(
            symbol=position.symbol, 
            instrument_type=position.instrument_type
        )

        linear_pricer = LinearPricer(
            position=position,
            currency_converter=self.currency_converter
        )

        if position.instrument_type == "ETF":
            return linear_pricer.etf_pricing(metadata=metadata)

        if position.instrument_type == "Future":
            return linear_pricer.future_pricing(metadata=metadata)

        if position.instrument_type == "FXSpot":
            return linear_pricer.fx_pricing(metadata=metadata)

        if position.instrument_type == "EuropeanOption":
            raise ValueError("EuropeanOption pricing not implemented yet.")

        raise ValueError(f"Unsupported instrument_type: {position.instrument_type}")