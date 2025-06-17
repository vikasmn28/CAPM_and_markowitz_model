import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import scipy.optimize as optimization
pd.set_option('display.max_columns', None)

Num_of_trading_days = 252
Num_of_portfolios = 10000

stocks = ['AAPL', 'WMT', 'TSLA', 'GE', 'AMZN', 'DB']

start_date = '2010-01-01'
end_date = '2017-01-01'


def download_data():
    stock_data = {}

    for stock in stocks:
        ticker = yf.Ticker(stock)
        stock_data[stock] = ticker.history(start = start_date, end = end_date)['Close']

    return pd.DataFrame(stock_data)

def show_data(data):
    data.plot(figsize=(10,5))
    plt.show()

def calculate_return(data):
    #To measure all values in comparable metric(We use logarithmic return instead of standard return)
    log_return = np.log(data/data.shift(1))
    return log_return[1:]

def show_statastics(returns):
    print(returns.mean() * Num_of_trading_days)
    print(returns.cov() * Num_of_trading_days)

def show_mean_varience(returns, weights):
    portfolio_return = np.sum(returns.mean() * weights) * Num_of_trading_days
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * Num_of_trading_days, weights)))
    print('Expected portfolio mean(returns): ',portfolio_return)
    print('Expected portfolio volatality(standard deviation): ',portfolio_volatility)

def show_porfolios(returns, volatilities):
    plt.figure(figsize=(10,6))
    plt.scatter(volatilities, returns, c=returns/volatilities, marker = 'o')
    plt.grid(True)
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected return')
    plt.colorbar(label='sharpe ratio')
    plt.show()

def Generate_portfolios(returns):
    portfolio_means = []
    portfolio_risk = []
    portfolio_weights = []
    for _ in range(Num_of_portfolios):
        w = np.random.random(len(stocks))
        w /= np.sum(w)
        portfolio_weights.append(w)
        portfolio_means.append(np.sum(returns.mean() * w) * Num_of_trading_days)
        portfolio_risk.append(np.sqrt(np.dot(w.T, np.dot(returns.cov() * Num_of_trading_days, w))))

    return np.array(portfolio_weights), np.array(portfolio_means), np.array(portfolio_risk)

def statistics(weights, returns)

if __name__ == '__main__':

  dataset = download_data()
  show_data(dataset)
  log_daily_return = calculate_return(dataset)
  show_statastics(log_daily_return)

  weights, means, risk = Generate_portfolios(log_daily_return)
  show_porfolios( means, risk)