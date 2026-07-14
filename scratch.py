from datetime import datetime
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

nyse = mcal.get_calendar("NYSE")

today_str = datetime.today().strftime("%Y-%m-%d")
expiry_str = expiry

valid_trading_days = nyse.valid_days(today_str, expiry_str).tz_localize(None)

if BUS:
    # BUS/252 (Trading Days / 252)
    # Best if volatility is scaled to a 252-day year
    trading_days_count = len(valid_trading_days) - 1
    year_fraction_bus252 = trading_days_count / 252.0

if ACT:
    # ACT/365 (Calendar Days / 365)
    # Best if interest rate (r) is scaled to a 365-day year
    t_start = pd.to_datetime(today_str)
    t_end = pd.to_datetime(expiry_str)
    calendar_days_count = (t_end - t_start).days
    year_fraction_act365 = calendar_days_count / 365.0


