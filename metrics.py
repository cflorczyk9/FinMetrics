import math
import pandas as pd
import numpy as np
from datetime import date
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib import style
import statistics

# User Inputs
years = 1 # make this a user input maybe?
end_date = date.today()
start_date = date.today() - pd.DateOffset(years=years) # make this a maximum number of years
risk_free_rate = yf.download("^TNX", start_date, end_date)['Adj Close'][0] / 100
ticker_id = "AAPL"
benchmark_id = "SPY"


def prepare_data_stock():
    tickers = [ticker_id]
    stock_data = yf.download(tickers, start_date, end_date)
    return stock_data


def prepare_data_benchmark():
    benchmark = [benchmark_id]
    benchmark_data = yf.download(benchmark, start_date, end_date)
    return benchmark_data


def prepare_data_ALL():
    ALL_tickers = [ticker_id, benchmark_id]
    ALL_data = yf.download(ALL_tickers, start_date, end_date)
    return ALL_data


def get_CAGR_stock():
    df = prepare_data_stock()
    df['daily_returns'] = df['Adj Close'].pct_change()
    df['cumulative_returns'] = (1 + df['daily_returns']).cumprod()
    trading_days = 252
    n = len(df)/ trading_days
    cagr_stock = round((df['cumulative_returns'][-1])**(1/n) - 1,2)
    return cagr_stock


def get_CAGR_benchmark():
    df = prepare_data_stock()
    df['daily_returns'] = df['Adj Close'].pct_change()
    df['cumulative_returns'] = (1 + df['daily_returns']).cumprod()
    trading_days = 252
    n = len(df)/ trading_days
    cagr_benchmark = round((df['cumulative_returns'][-1])**(1/n) - 1,2)
    return cagr_benchmark


def get_volatility():
    df = prepare_data_stock()
    df['daily_returns'] = df['Adj Close'].pct_change()
    trading_days = 252
    vol = round(df['daily_returns'].std() * np.sqrt(trading_days),2)
    return vol


def get_sharpe_ratio():
    sharpe = (get_CAGR_stock() - risk_free_rate)/ get_volatility()
    return sharpe


def get_correlation():
    df = prepare_data_ALL()
    df = df['Adj Close'].pct_change()
    beta = df.corr(method='pearson').iloc[1, 0]
    return beta


def get_alpha():
    alpha = get_CAGR_stock() - risk_free_rate - (get_correlation() * (get_CAGR_benchmark() - risk_free_rate))
    return alpha


def get_sortino_ratio():
    df = prepare_data_stock()
    df['daily_returns'] = df['Adj Close'].pct_change()
    df["negative_returns"] = np.where(df["daily_returns"]<0,df["daily_returns"],0)
    negative_volatility = df['negative_returns'].std() * np.sqrt(252)
    sortino = (get_CAGR_stock() - risk_free_rate)/ negative_volatility
    return sortino


def get_maximum_drawdown():
    df = prepare_data_stock()
    df['daily_returns'] = df['Adj Close'].pct_change()
    df['cumulative_returns'] = (1 + df['daily_returns']).cumprod()
    df['cumulative_max'] = df['cumulative_returns'].cummax()
    df['drawdown'] = df['cumulative_max'] - df['cumulative_returns']
    df['drawdown_pct'] = df['drawdown'] / df['cumulative_max']
    max_dd = df['drawdown_pct'].max()
    return max_dd


def get_calmar_ratio():
    calmar = (get_CAGR_stock() - risk_free_rate) / get_maximum_drawdown()
    return calmar


def get_VaR_90():
    df = prepare_data_stock()
    df['daily_returns'] = df['Adj Close'].pct_change()
    df = df['daily_returns']
    df = df.sort_values(ascending=True)
    VaR_90 = df.quantile(0.1)
    return VaR_90


def get_VaR_95():
    df = prepare_data_stock()
    df['daily_returns'] = df['Adj Close'].pct_change()
    df = df['daily_returns']
    df = df.sort_values(ascending=True)
    VaR_95 = df.quantile(0.05)
    return VaR_95


def get_VaR_99():
    df = prepare_data_stock()
    df['daily_returns'] = df['Adj Close'].pct_change()
    df = df['daily_returns']
    df = df.sort_values(ascending=True)
    VaR_99 = df.quantile(0.01)
    return VaR_99

def historical_chart():
    ticker_returns = prepare_data_stock()['Adj Close'].pct_change()
    benchmark_returns = prepare_data_benchmark()['Adj Close'].pct_change()

    ticker_return_cumulative = (ticker_returns + 1).cumprod()
    benchmark_return_cumulative = (benchmark_returns + 1).cumprod()

    last_price = prepare_data_stock()['Adj Close'][-1]

    fig = plt.figure()
    plt.plot(ticker_return_cumulative, label=ticker_id)
    plt.plot(benchmark_return_cumulative, label=benchmark_id)
    fig.autofmt_xdate()
    plt.legend()
    plt.show()


def monte_carlo_simulation():
    ticker_returns = prepare_data_stock()['Adj Close'].pct_change()
    benchmark_returns = prepare_data_benchmark()['Adj Close'].pct_change()

    num_simulations = 1000
    num_days = 252
    last_price = prepare_data_stock()['Adj Close'][-1]
    simulation_df = pd.DataFrame()

    for x in range(num_simulations):
        count = 0
        daily_vol = ticker_returns.std()

        price_series = []

        price = last_price * (1 + np.random.normal(0, daily_vol))
        price_series.append(price)

        for y in range(num_days):
            if count == 251:
                break
            price = price_series[count] * (1 + np.random.normal(0, daily_vol))
            price_series.append(price)
            count += 1

        simulation_df[x] = price_series

    style.use('seaborn-colorblind')

    fig = plt.figure()
    fig.suptitle(f'Monte Carlo Simulation: {ticker_id}')
    plt.plot(simulation_df)
    plt.axhline(y=last_price, color='black', linestyle='-')
    plt.xlabel('Day')
    plt.ylabel('Price')
    plt.show()


def normal_distribution():
    ticker_returns = prepare_data_stock()['Adj Close'].pct_change()
    benchmark_returns = prepare_data_benchmark()['Adj Close'].pct_change()

    num_simulations = 1000
    num_days = 252
    last_price = prepare_data_stock()['Adj Close'][-1]
    simulation_df = pd.DataFrame()

    for x in range(num_simulations):
        count = 0
        daily_vol = ticker_returns.std()

        price_series = []

        price = last_price * (1 + np.random.normal(0, daily_vol))
        price_series.append(price)

        for y in range(num_days):
            if count == 251:
                break
            price = price_series[count] * (1 + np.random.normal(0, daily_vol))
            price_series.append(price)
            count += 1

        simulation_df[x] = price_series

    plt.xlabel('Value')
    plt.ylabel('Number of simulation in each bin')
    plt.title(f'Normal Distribution: {ticker_id}')

    x = simulation_df.iloc[-1]

    s = x
    p = s.plot(kind='hist', bins=30, density=True, facecolor='g', alpha=0.75)

    bar_value_to_label = statistics.median(x)
    min_distance = float("inf")  # initialize min_distance with infinity
    index_of_bar_to_label = 0
    for i, rectangle in enumerate(p.patches):  # iterate over every bar
        tmp = abs(  # tmp = distance from middle of the bar to bar_value_to_label
            (rectangle.get_x() +
             (rectangle.get_width() * (1 / 2))) - bar_value_to_label)
        if tmp < min_distance:  # we are searching for the bar with x cordinate
            # closest to bar_value_to_label
            min_distance = tmp
            index_of_bar_to_label = i
    p.patches[index_of_bar_to_label].set_color('grey')
    # The colored bar is the Median
    plt.show()

print(normal_distribution())