from src.pricing.models import (
    PricingResult,
    Position
)

from src.instruments.models import (
    FutureMetadata,
    FXMetadata,
    ETFMetadata,
)

from src.pricing.currency_conversion import CurrencyConverter


class LinearPricer:
    def __init__(self, position: Position, currency_converter: CurrencyConverter):
        self.position = position
        self.currency_converter = currency_converter

    def etf_pricing(self, metadata: ETFMetadata) -> PricingResult:
        market_value = self.position.quantity * self.position.latest_price

        if metadata.currency != self.position.currency:
            market_value = self.currency_converter.convert_to_base(
                            amount=market_value, 
                            from_currency=metadata.currency
                            )

        abs_exposure = abs(market_value)

        return PricingResult(
            market_value=market_value,
            abs_exposure=abs_exposure
        )
    
    def future_pricing(self, metadata: FutureMetadata) -> PricingResult:

        notional = self.position.quantity * self.position.latest_price * metadata.multiplier

        if metadata.currency != self.position.currency:
            notional = self.cc.convert_to_base(
                            amount=notional, 
                            from_currency=metadata.currency
                            )
            
        abs_exposure = abs(notional)

        return PricingResult(
            market_value=notional,
            abs_exposure=abs_exposure
        )

    def fx_pricing(self, metadata: FXMetadata) -> PricingResult:
        quote_value = self.position.quantity * self.position.latest_price
        
        if metadata.quote_currency == self.position.currency:
            market_value = quote_value
        
        else:
            market_value = self.currency_converter.convert_to_base(
                            amount=quote_value, 
                            from_currency=metadata.quote_currency
                            )
            
        abs_exposure = abs(market_value)

        return PricingResult(
            market_value=market_value,
            abs_exposure=abs_exposure
        )