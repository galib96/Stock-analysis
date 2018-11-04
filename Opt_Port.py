
# %%

import quandl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import style

quandl.ApiConfig.api_key = 'Yxu-RF4ZvVHSv5HSkwc3'

selected_stock = ['CNP', 'F', 'WMT', 'GE', 'TSLA',
            'MSFT', 'T', 'CHK', 'AAPL', 'GOOG']
hist_data = quandl.get_table('WIKI/PRICES', ticker=selected_stock,
                        qopts={'columns': ['date', 'ticker', 'adj_close']},
                        date={'gte': '2014-1-1', 'lte': '2018-9-30'}, paginate=True)

clean = hist_data.set_index('date')
table = clean.pivot(columns='ticker')

returns_daily = table.pct_change()
returns_annual = returns_daily.mean() * 250

covariance_daily = returns_daily.cov()
covariance_annual = covariance_daily * 250

port_returns = []
port_volatility = []
stock_weights = []
sharpe_ratio = []

number_of_assets = len(selected_stock)
number_of_portfolios = 100000

for portfolio in range(number_of_portfolios):
    weights = np.random.random(number_of_assets)
    weights /= np.sum(weights)
    returns = np.dot(weights, returns_annual)
    volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_annual, weights)))
    sharpe = returns / volatility
    sharpe_ratio.append(sharpe)
    port_returns.append(returns)
    port_volatility.append(volatility)
    stock_weights.append(weights)

portfolios = {'Returns': port_returns,
              'Volatility': port_volatility, 'Sharpe Ratio': sharpe_ratio}

for counter, name in enumerate(selected_stock):
    portfolios[name + ' weight'] = [weights[counter]
                                    for weight in stock_weights]

df = pd.DataFrame(portfolios)

column_orders = ['Returns', 'Volatility', 'Sharpe Ratio'] + \
    [stock+' weight' for stock in selected_stock]
df = df[column_orders]


minimum_volatility = df['Volatility'].min()
maximum_sharpe = df['Sharpe Ratio'].max()


maximum_sharpe_portfolio = df.loc[df['Sharpe Ratio'] == maximum_sharpe]
minimum_variance_portfolio = df.loc[df['Volatility'] == minimum_volatility]


style.use('seaborn')
plt.scatter(x=df['Volatility'], y=df['Returns'],
            c=df['Sharpe Ratio'], cmap='summer')
plt.scatter(x=maximum_sharpe_portfolio['Volatility'],
            y=maximum_sharpe_portfolio['Returns'], c='red', marker='D', s=100)
plt.scatter(x=minimum_variance_portfolio['Volatility'],
            y=minimum_variance_portfolio['Returns'], c='blue', marker='D', s=100)
plt.xlabel('Volatility or Deviation')
plt.ylabel('Expected Returns of Portfolios')
plt.title('Efficient Frontier for Optimized Portfolio')
plt.show()

print('Minimum Risky Portfolio: \n', minimum_variance_portfolio.T)

print('Maximum Sharpe Ratio Portfolio: \n', maximum_sharpe_portfolio.T)

mvp = {}
for s in selected_stock:
    for col in minimum_variance_portfolio:
        if col == s+' weight':
            mvp[col] = minimum_variance_portfolio[col]

mvp = pd.DataFrame(mvp)

style.use('seaborn')
plt.pie(mvp.transpose(), labels=selected_stock, autopct='%1.1f%%')
plt.axis('equal')
plt.title('Portfolio with minimum variance')
plt.show()

msp = {}
for s in selected_stock:
    for col in maximum_sharpe_portfolio:
        if col == s+' weight':
            msp[col] = maximum_sharpe_portfolio[col]

msp = pd.DataFrame(msp)

style.use('seaborn-pastel')
plt.pie(msp.transpose(), labels=selected_stock, autopct='%1.2f%%')
plt.axis('equal')
plt.title('Portfolio with maximum sharpe ratio')
plt.show()
