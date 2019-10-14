# Libraries
import pandas as pd
from pandas_datareader import data as pdr
from pandas_datareader._utils import RemoteDataError
from scipy import stats
from scipy.stats import norm
import numpy as np
import datetime
import matplotlib.pyplot as plt
import math


# Class which analyze 1 single index based on historical data
class Index:

    # Constructor
    def __init__(self, index_name, num_of_years):
        self.__index_name = index_name
        self.__num_of_years = num_of_years
        self.__index_values = pd.DataFrame()
        self.__index_returns = pd.DataFrame()

        self.__download_hist_data()
        self.__calculate_daily_returns()

    # Download historical data
    def __download_hist_data(self):
        # Determine the first and the last days of the period
        end_date = datetime.date.today()
        start_date = datetime.date(end_date.year - self.__num_of_years,
                                   end_date.month, end_date.day)

        # Download data from Yahoo Finance
        try:
            self.__index_values = pdr.get_data_yahoo(self.__index_name,
                                                     start=start_date,
                                                     end=end_date)
        except RemoteDataError:
            # handle error
            print 'Stock symbol "{}" is not valid'.format(self.__index_name)

        self.__index_values = self.__index_values['Adj Close']

        self.__index_values.dropna(inplace=True)

    def __calculate_daily_returns(self):
        self.__index_returns = self.__index_values.pct_change(1)
        # Drop first line with NA
        self.__index_returns.dropna(inplace=True)

    # Getters
    def get_index_name(self):
        return self.__index_name

    def get_num_of_years(self):
        return self.__num_of_years

    def get_index_values(self):
        return self.__index_values

    def get_index_returns(self):
        return self.__index_returns

    # Normality test for index returns
    def perform_normality_test(self):
        p_value = stats.normaltest(self.__index_returns)[1]
        if(p_value < 0.05):
            print("Distribution of returns is not normal")
        else:
            print("Distribution of returns could be normal")
        return p_value

    # Lognormality test for index values
    def perform_lognormality_test(self):
        p_value = stats.normaltest(np.log(self.__index_values))[1]
        if(p_value < 0.05):
            print("Distribution of values is not lognormal")
        else:
            print("Distribution of values could be lognormal")
        return p_value


# Class to test a probability of an event
class Probability_Test:

    def __init__(self, mu, std, event):
        self.__mu = mu
        self.__std = std
        self.__event = event

    def get_mu(self):
        return self.__mu

    def get_std(self):
        return self.__std

    def get_event(self):
        return self.__event

    def check_probability(self):
        daily_st_dev = self.__std / math.sqrt(252)
        z_score = (self.__event - self.__mu)/daily_st_dev
        probability = (1 - stats.norm.cdf(z_score))*2
        print 'probability of the event is {} %'.format(probability*100)
        return probability


class Indices_Analysis:
    # Class constructor
    def __init__(self, indices_names, num_of_years):
        self.__indices = list()
        self.__indices_names = indices_names
        self.__num_of_years = num_of_years
        # Download data for all indices
        self.__download_indices()

    # Download indices data
    def __download_indices(self):
        for i in range(len(self.__indices_names)):
            index = Index(self.__indices_names[i], self.__num_of_years)
            self.__indices.append(index)

    def get_indices(self):
        return self.__indices

    def get_num_of_years(self):
        return self.__num_of_years

    def perform_normality_test(self):
        index = range(1)
        normality_test_results = pd.DataFrame(index=index,
                                              columns=self.__indices_names)
        for i in range(len(self.__indices)):
            index_name = self.__indices[i].get_index_name()
            p_value = self.__indices[i].perform_normality_test()
            normality_test_results[index_name][0] = p_value

        return normality_test_results

    def perform_lognormality_test(self):
        index = range(1)
        lognormality_test_results = pd.DataFrame(index=index,
                                                 columns=self.__indices_names)
        for i in range(len(self.__indices)):
            index_name = self.__indices[i].get_index_name()
            p_value = self.__indices[i].perform_lognormality_test()
            lognormality_test_results[index_name][0] = p_value

        return lognormality_test_results

    def plot_indices_values_distribution(self):
        fig, axes = plt.subplots(nrows=4, ncols=2, sharex=True, sharey=False,
                                 figsize=(12, 12))
        for i in range(4):
            for j in range(2):
                ts = self.__indices[2*i+j].get_index_values()

                # Statistics calculations
                log_ts = np.log(ts.dropna())
                mu, std = norm.fit(log_ts)
                kurtosis = stats.kurtosistest(log_ts).pvalue

                axes[i, j].axis('off')
                axes[i, j].set_title(self.__indices_names[2*i+j])
                axes[i, j].hist(log_ts.dropna(), bins=30, normed=True,
                                color='red')

                xmin, xmax = axes[i, j].get_xlim()
                x = np.linspace(xmin, xmax, 100)
                p = norm.pdf(x, mu, std)

                axes[i, j].fill_between(x, 0, p, color='grey', alpha='0.7')
                axes[i, j].plot(x, p, 'k', linewidth=2)

                title = "%s, mu=%.2f, sigma=%.2f, kurt_pv=%.2f" % (
                        self.__indices_names[2*i+j], mu, std, kurtosis)

                axes[i, j].set_title(title)
                plt.suptitle("Distribution of indices logarithm values ")

    def plot_indices_returns_distribution(self):
        fig, axes = plt.subplots(nrows=4, ncols=2, sharex=True, sharey=False,
                                 figsize=(12, 12))
        for i in range(4):
            for j in range(2):
                ts = self.__indices[2*i+j].get_index_returns()

                # Statistics calculations
                ts = ts.dropna()
                mu, std = norm.fit(ts)
                kurtosis = stats.kurtosistest(ts).pvalue

                axes[i, j].axis('off')
                axes[i, j].set_title(self.__indices_names[2*i+j])
                axes[i, j].hist(ts.dropna(), bins=30, normed=True,
                                color='red')

                xmin, xmax = axes[i, j].get_xlim()
                x = np.linspace(xmin, xmax, 100)
                p = norm.pdf(x, mu, std)

                axes[i, j].fill_between(x, 0, p, color='grey', alpha='0.7')
                axes[i, j].plot(x, p, 'k', linewidth=2)

                title = "%s, mu=%.2f, sigma=%.2f, kurt_pv=%.2f" % (
                        self.__indices_names[2*i+j], mu, std, kurtosis)

                axes[i, j].set_title(title)
                plt.suptitle("Distribution of indices returns ")


index_name = ['^DJI', '^GSPC', '^IXIC', '^FTSE', '^HSI', '^KS11',
              '^NSEI', '^GDAXI']
number_of_years = 25

indices = Indices_Analysis(index_name, number_of_years)
df1 = indices.perform_normality_test()
df2 = indices.perform_lognormality_test()
indices.plot_indices_values_distribution()
indices.plot_indices_returns_distribution()
test = Probability_Test(0, 0.2, 0.226)
test.check_probability()
