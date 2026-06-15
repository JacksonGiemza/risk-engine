import pandas as pd
import yfinance as yf

def get_price_history(portfolio_path, start_date, end_date):

    portfolio = pd.read_csv(portfolio_path)
    tickers = list(portfolio.symbol)

    history = yf.download(tickers, start=start_date, end=end_date,auto_adjust=True)

    adj_close_df = history['Close']

    adj_close_df.to_csv(rf'../data/port_ticker_history_{start_date}_{end_date}.csv')

def save_prices_to_csv():
    pass

def validate_prices():
    pass

def calculate_prices():
    pass