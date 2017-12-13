#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 13:17:29 2017

@author: mmundo, navarroj @mit.edu
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import statsmodels
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima_model import ARIMA




class ARMAmodels(object):
    
    def __init__(self):
        self.data=None
        self.ts=None
        self.model = None
        self.results = None
    
    def loadData(self, filename, datasize, sampling):
        self.data = pd.read_csv(filename)
             
        self.data[self.data.columns[0]] = pd.to_datetime(self.data[self.data.columns[0]], unit='s')
        self.data = self.data.tail(datasize)
        self.data = self.data.set_index(self.data.columns[0])
        self.ts=self.data[self.data.columns[1]]
        self.ts = self.ts.resample(sampling).mean()
        
        self.ts.head(5)
        plt.plot(self.ts.to_pydatetime(), self.ts.values)
        plt.show()
        return self
        
    def test_stationarity(self):
        #https://datascience.ibm.com/exchange/public/entry/view/815137c868b916821dec777bdc23013c
        #Determing rolling statistics
        timeseries = self.ts
        rolmean = timeseries.rolling(window=24,center=True).mean()
        rolstd = timeseries.rolling(window=24,center=True).std()
    
        #Plot rolling statistics:
        orig = plt.plot(timeseries.index.to_pydatetime(), timeseries.values, color='blue',label='Original')
        mean = plt.plot(rolmean.index.to_pydatetime(), rolmean.values, color='red', label='Rolling Mean')
        std = plt.plot(rolstd.index.to_pydatetime(), rolstd.values, color='black', label = 'Rolling Std')
        plt.legend(loc='best')
        plt.title('Rolling Mean & Standard Deviation')
        plt.show(block=False)
    
        #Perform Dickey-Fuller test:
        print('Results of Dickey-Fuller Test:')
        dftest = adfuller(timeseries, maxlag=10)
        dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
        for key,value in dftest[4].items():
            dfoutput['Critical Value (%s)'%key] = value
        print(dfoutput)
        return dfoutput
    
    def deTrending(self):
        timeseries = self.ts
        #if Dickey Fuller test fails try log transform
        #if test_stationary(timeseries)[1] > 0.05:
        timeseries_log = np.log(timeseries)
        #if Dickey Fuller test fails again try difftransform
        #if test_stationary(timeseries)[1] > 0.05:
        timeseries_log_diff = timeseries_log - timeseries_log.shift()
        self.ts = timeseries_log_diff.dropna(inplace=True)
        return self.ts
    
    def acf_pacf(self, nlags):
        lag_acf = acf(self.ts, nlags)
        lag_pacf = pacf(self.ts, nlags, method='ols')
        lineslope = 1.96
        plt.subplot(121)
        plt.plot(lag_acf)
        plt.axhline(y=0, linestyle='--', color='gray')
        plt.axhline(y=-lineslope/np.sqrt(len(ts_week_log_diff)),linestyle='--',color='gray')
        plt.axhline(y=lineslope/np.sqrt(len(ts_week_log_diff)),linestyle='--',color='gray')
        plt.title('Autocorrelation Function')

        plt.subplot(122)
        plt.plot(lag_pacf)
        plt.axhline(y = 0, linestyle='--', color ='gray')
        plt.axhline(y=-lineslope/np.sqrt(len(ts_week_log_diff)), linestyle='--', color='gray')
        plt.axhline(y=lineslope/np.sqrt(len(ts_week_log_diff)),linestyle='--', color='gray')
        plt.title('Partial Autocorrelation')

        plt.tight_layout()
        plt.show()
    
    def fit(self, model):
        if model =='ARMA':
            results_ARMA = ARMA(self.ts, order = (2,1,2)).fit()
            plt.plot(self.ts)
            plt.plot(results_ARMA.fittedvalues, color='red')
            plt.title('RSS: %.4f'% sum((results_ARMA.fittedvalues-self.ts)**2))
            print(results_ARMA.summary())
            return results_ARMA
        if model == 'ARIMA':
            results_ARMA = ARIMA(self.ts, order = (2,1,2)).fit()
            plt.plot(self.ts)
            plt.plot(results_ARMA.fittedvalues, color='red')
            plt.title('RSS: %.4f'% sum((results_ARMA.fittedvalues-self.ts)**2))
            
            print(results_ARMA.summary())
            return results_ARMA
    
    def predict(self):
        predictions_diff = pd.Series(self.fit('ARMA').fittedvalues, copy=True)
        predictions_diff_cumsum = predictions_diff.cumsum()
        predictions_log = pd.Series(self.ts.iloc[0], index=ts_week_log.index)
        predictions_log = predictions_log.add(predictions_diff_cumsum,fill_value=0)
        
        predictions_ARMA = np.exp(predictions_log)
        plt.plot(self.ts.index.to_pydatetime(), ts_week.values)
        plt.plot(self.ts.index.to_pydatetime(), predictions_ARMA.values)
        plt.title('ARMA predictions against real values')
        plt.xlabel('Dates')
        plt.ylabel('Price (BTC/USD$)') 
        print('RMSE: %.4f' % np.sqrt((ts_week.add(-predictions_ARMA)**2).sum()/len(ts_week)))