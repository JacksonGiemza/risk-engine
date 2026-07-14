from datetime import datetime
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

from src.instruments.models import OptionMetadata

# https://www.codearmo.com/python-tutorial/options-trading-black-scholes-model

class OptionPricer:
    def __init__(self):
        pass


    def black_scholes(self, metadata: OptionMetadata, asset_returns):

        T = self._get_time_to_expiry(metadata.expiry)




    
    def _get_time_to_expiry(self, expiry: str, BUS=True, ACT=False) -> float:
        nyse = mcal.get_calendar("NYSE")

        today_str = datetime.today().strftime("%Y-%m-%d")
        expiry_str = expiry

        valid_trading_days = nyse.valid_days(today_str, expiry_str).tz_localize(None)

        if BUS:
            # BUS/252 (Trading Days / 252)
            # Best if volatility is scaled to a 252-day year
            trading_days_count = len(valid_trading_days) - 1
            T = trading_days_count / 252.0
            return T
        
        if ACT:
            # ACT/365 (Calendar Days / 365)
            # Best if interest rate (r) is scaled to a 365-day year
            t_start = pd.to_datetime(today_str)
            t_end = pd.to_datetime(expiry_str)
            calendar_days_count = (t_end - t_start).days
            T = calendar_days_count / 365.0
            return T 