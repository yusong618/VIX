#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 15:39:54 2019

@author: yu
"""

import pandas as pd  
import numpy as np
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
yf.pdr_override() 
import datetime
import matplotlib.pyplot as plt


vix = pd.read_excel('VIX.xlsx', header=0, index_col=0)

startdate=datetime.datetime(1993,1,2)
enddate=datetime.datetime(2019,7,13)
sp500 = pdr.get_data_yahoo('^GSPC', start=startdate, end=enddate)

sp500_close = sp500['Adj Close']

data = pd.DataFrame(index=sp500_close.index)
data['vix']=vix
data['sp500']=sp500_close
data['vix/sp500']=data['vix']/data['sp500']
data['sp500/vix']=data['sp500']/data['vix']

startdate1 = datetime.datetime(2016,1,1)
data1 = data.loc[startdate1: enddate,:]


fig,ax1=plt.subplots(figsize=(12,8))
ax2=ax1.twinx()
l1, = ax1.plot(-data1['vix'],'r--')
l2, = ax2.plot(data1['sp500'], 'g-')
ax1.set_ylabel('-vix', color  = 'r')
ax2.set_ylabel('sp500', color = 'g')




fig,ax1=plt.subplots(figsize=(12,8))
ax2=ax1.twinx()
l1, = ax1.plot(data['sp500/vix'],'r--')
l2, = ax2.plot(data['sp500'], 'g-')
ax1.set_ylabel('sp500/vix',color = 'r')
ax2.set_ylabel('sp500', color = 'g')


corr=data.corr()

data['sp500_ret']=data['sp500'].pct_change()
data = data.dropna()

# 单位根检验
from arch.unitroot import ADF
ADF(data['sp500'])
ADF(data['vix/sp500'])
# 结果：两个时间序列都不平稳


#协整性检验(原始数据)
from statsmodels.tsa.stattools import coint
print(coint(data['sp500'], data['vix/sp500']))
# 结果：不存在协整性
 
# 一阶单整检验
from statsmodels.tsa.stattools import adfuller
sp500_1 = np.reshape(data['sp500'].values, -1)
sp500_1 = np.diff(sp500_1)
 
vix_sp500_1 = np.reshape(data['vix/sp500'].values, -1)
vix_sp500_1 = np.diff(vix_sp500_1)
 
print(adfuller(sp500_1))
print(adfuller(vix_sp500_1))
# 结果：一阶差分后的数据满足一阶单整

#协整性检验（一阶差分后的数据）
print(coint(sp500_1,vix_sp500_1))
# 结果：一阶差分后的数据存在协整性，可以进行因果关系检验

#格兰杰因果关系检验
from statsmodels.tsa.stattools import grangercausalitytests
data_granger=pd.DataFrame(index=vix.index)
data_granger['sp500']=data['sp500']
data_granger['vix/sp500']=data['vix/sp500']
data_granger = data_granger.dropna()
grangercausalitytests(data_granger[['sp500', 'vix/sp500']], maxlag=7)


#%%
# OLS回归
lag=3
data_ols=data_granger.dropna()
data_ols['vix/sp500']=data_ols['vix/sp500']*100
data_ols['vix/sp500']=data_ols['vix/sp500'].shift(lag)
data_ols=data_ols.dropna()

import statsmodels.api as sm
features = data_ols['vix/sp500']
linear_features = sm.add_constant(features)

size=len(data_ols)
train_size=int(0.7*size)
train_targets = data_ols.iloc[:train_size,0]
train_features = linear_features[:train_size]
test_targets = data_ols.iloc[train_size:,0]
test_features = linear_features[train_size:]

model = sm.OLS(train_targets, train_features)
results = model.fit()

print(results.summary())


# 回归结果：R—squared = 0.462

train_predictions = results.predict(train_features)
test_predictions = results.predict(test_features)

plt.scatter(test_predictions, test_targets, alpha = 0.2, color = 'r', label = 'test')
plt.scatter(train_predictions, train_targets, alpha = 0.2, color = 'b', label = 'train')

xmin, xmax = plt.xlim()
plt.plot(np.arange(xmin, xmax, 0.01), np.arange(xmin, xmax, 0.01), c = 'k')

plt.xlabel('predictions')
plt.ylabel('targets')

#%%
# 策略

train = data_ols.iloc[:train_size]
test = data_ols.iloc[train_size:]

vix_train_mean = train.iloc[:,1].mean()
vix_train_std = train.iloc[:,1].std()

sell_signal = vix_train_mean + vix_train_std/2
buy_signal = vix_train_mean - vix_train_std/2

fig,ax1 = plt.subplots(figsize = (14,7))
ax2 = ax1.twinx()
l1, = ax1.plot(train.iloc[:,0], color = 'yellow')
l2, = ax2.plot(train.iloc[:,1])
ax1.set_ylabel('sp500')
ax2.set_xlabel('vix/sp500')
plt.axhline(buy_signal, color = 'r', lw = 3)
plt.axhline(vix_train_mean, color = 'black', lw = 1)
plt.axhline(sell_signal, color = 'g', lw = 3)
plt.legend(['sp500_train', 'buy_signal', 'vix_train_mean', 'sell_signal'], loc = 'best')
plt.show()

fig,ax1 = plt.subplots(figsize = (14,7))
ax2 = ax1.twinx()
l1, = ax1.plot(test.iloc[:,0], color = 'yellow')
l2, = ax2.plot(test.iloc[:,1])
ax1.set_ylabel('sp500')
ax2.set_xlabel('vix/sp500')
plt.axhline(buy_signal, color = 'r', lw = 3)
plt.axhline(vix_train_mean, color = 'black', lw = 1)
plt.axhline(sell_signal, color = 'g', lw = 3)
plt.legend(['sp500_test', 'buy_signal', 'vix_train_mean', 'sell_signal'], loc = 'best')
plt.show()



buy_index = test[test['vix/sp500']<=buy_signal].index
test.loc[buy_index,'signal']=1
sell_index = test[test['vix/sp500']>sell_signal].index
test.loc[sell_index,'signal']=0

test['keep'] = test['signal']
test['keep'].fillna(method = 'ffill', inplace = True)

test['benchmark_profit'] = np.log(test['sp500']/test['sp500'].shift(1))
test['trend_profit'] = test['keep'] * test['benchmark_profit']

test[['benchmark_profit', 'trend_profit']].cumsum().plot(grid=True, figsize=(14,7))


#%%
data1=pd.DataFrame(index=vix.index)
data1['sp500']=data['sp500']
data1['vix/sp500']=data['vix/sp500']

criteria = pd.DataFrame(index=data1.index)
criteria['mean']=np.nan
criteria['std'] = np.nan

for i in range(0,len(data1)):
    train = data1.iloc[i:i+252,:]
    if i >= 252:
        criteria.iloc[i,0] = train.iloc[:,1].mean()
        criteria.iloc[i,1] = train.iloc[:,1].std()

criteria['buy_signal'] = criteria['mean'] + criteria['std']/2
criteria['sell_signal'] = criteria['mean'] - criteria['std']/2

criteria['vix/sp500'] = data1['vix/sp500']

buy_index = criteria[criteria['vix/sp500']<=criteria['buy_signal']].index
criteria.loc[buy_index, 'signal'] = 1
sell_index = criteria[criteria['vix/sp500']>criteria['sell_signal']].index
criteria.loc[sell_index, 'signal'] = 0

criteria['keep'] = criteria['signal']
criteria['keep'].fillna(method = 'ffill', inplace = True)

criteria['sp500'] = data1['sp500']
criteria.dropna()
criteria['benchmark_profit'] = np.log(criteria['sp500']/criteria['sp500'].shift(1))
criteria['trend_profit'] = criteria['keep'] * criteria['benchmark_profit']

criteria[['benchmark_profit','trend_profit']].cumsum().plot(grid=True, figsize=(14,7))
