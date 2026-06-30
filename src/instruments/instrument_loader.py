from pathlib import Path
import polars as pl

from models import (
    FutureMetaData, 
    EtfMetaData, 
    FxMetaData, 
    OptionsMetaData)


class InstrumentLoader:
    def __init__(self):
        self.root = Path(__file__).resolve().parent / "instrument_metadata"

        self.etfs = pl.read_csv(self.root / "etfs.csv")
        self.futures = pl.read_csv(self.root / "futures.csv")
        self.fx = pl.read_csv(self.root / "fx.csv")
        self.options = pl.read_csv(self.root / "options.csv")
    
    def load(self, symbol):
        pass
    
    def load_etfs(self, symbol):
        PATH = self.ROOT / "etfs.csv"
        etf = pl.scan_csv(PATH).filter(
            pl.col("symbol") == symbol
            ).collect()
        
        return EtfMetaData(
            symbol=etf['symbol'][0],
            expense_ratio=etf['expense_ratio'][0],
            issuer=etf['issuer'][0],
            asset_class=etf['asset_class'][0]
        )

    def load_futures(self, symbol):
        PATH = self.ROOT / "futures.csv"
        future = pl.scan_csv(PATH).filter(
            pl.col("symbol") == symbol
            ).collect()
        
        return FutureMetaData(
            symbol=future['symbol'][0],
            multiplier=future['multiplier'][0],
            exchange=future['exchange'][0],
            asset_class=future['asset_class'][0]
        )
    
    def load_fx(self, symbol):
        PATH = self.ROOT / "fx.csv"
        fx = pl.scan_csv(PATH).filter(
                    pl.col("symbol") == symbol
                    ).collect()
                
        return FxMetaData(
            symbol=fx['symbol'][0],
            base_currency=fx['base_currency'][0],
            quote_currency=fx['quote_currency'][0],
            asset_class=fx['asset_class'][0]
        )
    
    def load_options(self, symbol):
        PATH = self.ROOT / "options.csv"
        option = pl.scan_csv(PATH).filter(
            pl.col("symbol") == symbol
            ).collect()
        
        return OptionsMetaData(
            symbol=option['symbol'][0],
            strike=option['strike'][0],
            expiry=option['expiry'][0],
            multiplier=option['multiplier'][0],
            option_type=option['option_type'][0],
            exercise_style=option['exercise_style'][0],
            currency=option['currency'][0]
        )
    
    def _lookup(self, metadata: pl.DataFrame, symbol: str, instrument_type: str) -> dict:
        row = metadata.filter(pl.col("symbol") == symbol)

        if row.is_empty():
            raise ValueError(f"No {instrument_type} metadata found for symbol: {symbol}")

        if row.height > 1:
            raise ValueError(f"Multiple {instrument_type} metadata rows found for symbol: {symbol}")

        return row.row(0, named=True)
    

def main():
    pass    
if __name__ == "__main__":
    main()
