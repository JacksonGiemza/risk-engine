import datetime

from dateutil.relativedelta import relativedelta
from src.instruments.models import OptionMetadata

class OptionPricer:
    def __init__(self):
        pass


    def black_scholes(self, metadata: OptionMetadata, asset_returns):
        expiry = datetime.strptime(metadata.expiry, "%Y-%m-%d").date()
        today = datetime.today().date()

        T = relativedelta(metadata.expiry, today).years

        S = asset_returns[metadata.underlying].iloc[0]

        sigma = 

        # https://www.codearmo.com/python-tutorial/options-trading-black-scholes-model