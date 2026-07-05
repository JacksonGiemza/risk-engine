from src.pricing.models import (
    PricingResult,
    Position
)

from src.instruments.models import (
    FutureMetadata,
    FXMetadata,
    OptionMetadata
)


class LinearPricing:
    def __init__(self, position: Position):
        self.position = position

    def etf_pricing(self):
        market_value = self.position.quantity * self.position.latest_price
        abs_exposure = abs(market_value)

        return PricingResult(
            market_value=market_value,
            abs_exposure=abs_exposure
        )
    
    def future_pricing(self, metadata: FutureMetadata):
        market_value = self.position.quantity * self.position.latest_price * metadata.multiplier
        abs_exposure = abs(market_value)

        return PricingResult(
            market_value=market_value,
            abs_exposure=abs_exposure
        )

    def fx_pricing(self, metadata: FXMetadata):
        quote_value = self.position.quantity * self.position.latest_price
        
        if metadata.quote_currency == self.position.currency:
            market_value = quote_value
        
        else:
            market_value = 