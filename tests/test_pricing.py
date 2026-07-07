import pytest

from src.pricing.linear_pricers import LinearPricer
from src.pricing.models import Position, PricingResult
from src.instruments.models import ETFMetadata, FutureMetadata, FXMetadata


class MockCurrencyConverter:
    def __init__(self, rates=None):
        self.rates = rates or {}

    def convert_to_base(self, amount: float, from_currency: str) -> float:
        rate = self.rates.get(from_currency)

        if rate is None:
            raise ValueError(f"No mock rate for {from_currency}")

        return amount / rate


def test_etf_pricing_usd():
    position = Position(
        symbol="SPY",
        instrument_type="ETF",
        asset_class="Equity",
        quantity=100,
        currency="USD",
        market_price=500,
    )

    metadata = ETFMetadata(
        symbol="SPY",
        instrument_type="ETF",
        asset_class="Equity",
        currency="USD",
        expense_ratio=0.0945,
        issuer="State Street",
    )

    pricer = LinearPricer(position, MockCurrencyConverter())
    result = pricer.etf_pricing(metadata)

    assert isinstance(result, PricingResult)
    assert result.market_value == pytest.approx(50_000)
    assert result.abs_exposure == pytest.approx(50_000)

def test_etf_pricing_foreign_currency_converts_to_base():
    position = Position(
        symbol="EXAMPLE_EUR_ETF",
        instrument_type="ETF",
        asset_class="Equity",
        quantity=100,
        currency="USD",
        market_price=50,
    )

    metadata = ETFMetadata(
        symbol="EXAMPLE_EUR_ETF",
        instrument_type="ETF",
        asset_class="Equity",
        currency="EUR",
        expense_ratio=0.20,
        issuer="Example Issuer",
    )

    converter = MockCurrencyConverter(rates={"EUR": 0.80})

    pricer = LinearPricer(position, converter)
    result = pricer.etf_pricing(metadata)

    assert result.market_value == pytest.approx(6_250)
    assert result.abs_exposure == pytest.approx(6_250)


def test_future_pricing_usd():
    position = Position(
        symbol="ES=F",
        instrument_type="Future",
        asset_class="Equity",
        quantity=2,
        currency="USD",
        market_price=6_000,
    )

    metadata = FutureMetadata(
        symbol="ES=F",
        instrument_type="Future",
        asset_class="Equity",
        currency="USD",
        multiplier=50,
        exchange="CME",
    )

    pricer = LinearPricer(position, MockCurrencyConverter())
    result = pricer.future_pricing(metadata)

    assert result.market_value == pytest.approx(600_000)
    assert result.abs_exposure == pytest.approx(600_000)


def test_short_future_pricing_abs_exposure_positive():
    position = Position(
        symbol="CL=F",
        instrument_type="Future",
        asset_class="Commodity",
        quantity=-2,
        currency="USD",
        market_price=75,
    )

    metadata = FutureMetadata(
        symbol="CL=F",
        instrument_type="Future",
        asset_class="Commodity",
        currency="USD",
        multiplier=1000,
        exchange="NYMEX",
    )

    pricer = LinearPricer(position, MockCurrencyConverter())
    result = pricer.future_pricing(metadata)

    assert result.market_value == pytest.approx(-150_000)
    assert result.abs_exposure == pytest.approx(150_000)


def test_future_pricing_foreign_currency_converts_to_base():
    position = Position(
        symbol="FDAX",
        instrument_type="Future",
        asset_class="Equity",
        quantity=2,
        currency="USD",
        market_price=25_000,
    )

    metadata = FutureMetadata(
        symbol="FDAX",
        instrument_type="Future",
        asset_class="Equity",
        currency="EUR",
        multiplier=25,
        exchange="EUREX",
    )

    converter = MockCurrencyConverter(rates={"EUR": 0.80})

    pricer = LinearPricer(position, converter)
    result = pricer.future_pricing(metadata)

    assert result.market_value == pytest.approx(1_562_500)
    assert result.abs_exposure == pytest.approx(1_562_500)


def test_fx_pricing_quote_currency_matches_portfolio_currency():
    position = Position(
        symbol="EURUSD=X",
        instrument_type="FXSpot",
        asset_class="FX",
        quantity=50_000,
        currency="USD",
        market_price=1.10,
    )

    metadata = FXMetadata(
        symbol="EURUSD=X",
        instrument_type="FXSpot",
        asset_class="FX",
        base_currency="EUR",
        quote_currency="USD",
    )

    pricer = LinearPricer(position, MockCurrencyConverter())
    result = pricer.fx_pricing(metadata)

    assert result.market_value == pytest.approx(55_000)
    assert result.abs_exposure == pytest.approx(55_000)


def test_fx_pricing_quote_currency_converts_to_base():
    position = Position(
        symbol="USDJPY=X",
        instrument_type="FXSpot",
        asset_class="FX",
        quantity=50_000,
        currency="USD",
        market_price=150,
    )

    metadata = FXMetadata(
        symbol="USDJPY=X",
        instrument_type="FXSpot",
        asset_class="FX",
        base_currency="USD",
        quote_currency="JPY",
    )

    converter = MockCurrencyConverter(rates={"JPY": 150})

    pricer = LinearPricer(position, converter)
    result = pricer.fx_pricing(metadata)

    assert result.market_value == pytest.approx(50_000)
    assert result.abs_exposure == pytest.approx(50_000)


def test_short_fx_position_abs_exposure_positive():
    position = Position(
        symbol="EURUSD=X",
        instrument_type="FXSpot",
        asset_class="FX",
        quantity=-50_000,
        currency="USD",
        market_price=1.10,
    )

    metadata = FXMetadata(
        symbol="EURUSD=X",
        instrument_type="FXSpot",
        asset_class="FX",
        base_currency="EUR",
        quote_currency="USD",
    )

    pricer = LinearPricer(position, MockCurrencyConverter())
    result = pricer.fx_pricing(metadata)

    assert result.market_value == pytest.approx(-55_000)
    assert result.abs_exposure == pytest.approx(55_000)
    