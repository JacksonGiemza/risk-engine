from pathlib import Path
from typing import TypeAlias

import polars as pl

from src.instruments.models import (
    InstrumentMetaData,
    FutureMetaData,
    EtfMetaData,
    FxMetaData,
    OptionsMetaData,
)


Instrument: TypeAlias = (
    InstrumentMetaData
    | FutureMetaData
    | EtfMetaData
    | FxMetaData
    | OptionsMetaData
)


class InstrumentLoader:
    def __init__(self) -> None:
        self.root: Path = Path(__file__).resolve().parent / "instrument_metadata"

        self.etfs: pl.DataFrame = pl.read_csv(self.root / "etfs.csv")
        self.futures: pl.DataFrame = pl.read_csv(self.root / "futures.csv")
        self.fx: pl.DataFrame = pl.read_csv(self.root / "fx.csv")
        self.options: pl.DataFrame = pl.read_csv(self.root / "options.csv")

    def load(self, symbol: str, instrument_type: str) -> Instrument:
        if instrument_type == "ETF":
            return self.load_etfs(symbol)

        if instrument_type == "Future":
            return self.load_futures(symbol)

        if instrument_type == "FXSpot":
            return self.load_fx(symbol)

        if instrument_type == "EuropeanOption":
            return self.load_options(symbol)

        raise ValueError(f"Unsupported instrument_type: {instrument_type}")

    def load_etfs(self, symbol: str) -> EtfMetaData:
        etf = self._lookup(self.etfs, symbol, "ETF")

        return EtfMetaData(
            symbol=etf["symbol"],
            instrument_type="ETF",
            asset_class=etf["asset_class"],
            expense_ratio=float(etf["expense_ratio"]),
            issuer=etf["issuer"],
        )

    def load_futures(self, symbol: str) -> FutureMetaData:
        future = self._lookup(self.futures, symbol, "Future")

        return FutureMetaData(
            symbol=future["symbol"],
            instrument_type="Future",
            asset_class=future["asset_class"],
            multiplier=float(future["contract_multiplier"]),
            exchange=future["exchange"],
        )

    def load_fx(self, symbol: str) -> FxMetaData:
        fx = self._lookup(self.fx, symbol, "FXSpot")

        return FxMetaData(
            symbol=fx["symbol"],
            instrument_type="FXSpot",
            asset_class=fx["asset_class"],
            base_currency=fx["base_currency"],
            quote_currency=fx["quote_currency"],
        )

    def load_options(self, symbol: str) -> OptionsMetaData:
        option = self._lookup(self.options, symbol, "EuropeanOption")

        return OptionsMetaData(
            symbol=option["symbol"],
            instrument_type="EuropeanOption",
            asset_class=option.get("asset_class", "Equity"),
            strike=float(option["strike"]),
            expiry=option["expiry"],
            multiplier=float(option["contract_multiplier"]),
            option_type=option["option_type"],
            exercise_style=option["exercise_style"],
        )

    def _lookup(
        self,
        metadata: pl.DataFrame,
        symbol: str,
        instrument_type: str,
    ) -> dict:
        row = metadata.filter(pl.col("symbol") == symbol)

        if row.is_empty():
            raise ValueError(f"No {instrument_type} metadata found for symbol: {symbol}")

        if row.height > 1:
            raise ValueError(
                f"Multiple {instrument_type} metadata rows found for symbol: {symbol}"
            )

        return row.row(0, named=True)