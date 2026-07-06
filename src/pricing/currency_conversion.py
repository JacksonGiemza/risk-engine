import requests
import os
from dotenv import load_dotenv

load_dotenv()

class CurrencyConverter:
    def __init__(self, portfolio_currency ="USD"):
        self.API_KEY = os.getenv("EXCHANGE_RATE_KEY")
        self.portfolio_currency = portfolio_currency 
        self.url = f'https://v6.exchangerate-api.com/v6/{self.API_KEY}/latest/'
        self.rates = {}

        if not self.API_KEY:
            raise ValueError("API key missing. Check your .env file.")

    def fetch_rates(self):
        try:
            response = requests.get(f"{self.url}{self.portfolio_currency}")
            response.raise_for_status()
            data = response.json()
            self.rates = data.get("conversion_rates", {})

        except requests.RequestException as e:
            raise ValueError(f"Error fetching exchange rates: {e}")

    def convert_to_base(self, amount, from_currency):
        if not self.rates:
            self.fetch_rates()

        if from_currency == self.portfolio_currency:
            return amount

        rate = self.rates.get(from_currency)
        if not rate:
            raise ValueError(f"Currency code '{from_currency}' not supported or found.")
            
        return amount / rate        
