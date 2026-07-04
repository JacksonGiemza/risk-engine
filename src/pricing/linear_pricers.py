from src.pricing.models import (
    PricingResult,
    Position
)

class LinearPricing:
    def __init__(self, position: Position) -> PricingResult:
        self.position = position


    def etf_pricing(self):
        market_value = self.position.quantity * self.position.latest_price
        abs_exposure = abs(market_value)

        return PricingResult(
            market_value=market_value,
            abs_exposure=abs_exposure
        )
    
    