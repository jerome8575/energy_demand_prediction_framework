
# librairies
import sys
sys.path.insert(0, "C:\\Users\\jerom\\coding\\energy_demand_prediction_framework\\")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from data.quebec_energy_demand import HQ_data
from models.spline_regression import SplineRegression
from statsmodels.tools.eval_measures import rmse

class Simulation:

    def __init__(self, num_iters, train_start, train_end, test_start, test_end):

        self.num_iters = num_iters
        self.train_start = train_start
        self.train_end = train_end
        self.test_start = test_start
        self.test_end = test_end
        self.data = HQ_data()
        self.data = self.data.get_history()

    def get_prediction(self, train_start, train_end, test_start, test_end):

        """ implement algorithm or call algorithm here. Return array of 24 values for next day forecast """
        sr = SplineRegression()
        return sr.get_predictions(self.data, train_start, train_end, test_start, test_end)

    def run_simulation(self):
        
        train_start = self.train_start
        train_end = self.train_end
        test_start = self.test_start
        test_end = self.test_end

        forecasts = []

        for i in range(self.num_iters):

            forecast = self.get_prediction(train_start, train_end, test_start, test_end)
            
            forecasts.append(forecast)

            print("********************************************")
            print("At iteration"+ str(i))
            print("forecasts: ", forecast)
            print("********************************************")

            train_start = train_start + datetime.timedelta(days=1)
            train_end = train_end + datetime.timedelta(days=1)
            test_start = test_start + datetime.timedelta(days=1)
            test_end = test_end + datetime.timedelta(days=1)

        return np.array(forecasts).flatten()

    def plot_sim_results(self, forecasts):

        sim_start = self.train_end + datetime.timedelta(hours=1)
        sim_end = sim_start + datetime.timedelta(days = self.num_iters) - datetime.timedelta(hours=1)

        results = self.data.loc[sim_start:sim_end, ["demand"]]

        results["forecast"] = forecasts

        demand = np.array(self.data.loc[sim_start:sim_end, "demand"])

        MSE = rmse(forecasts, demand)
        print("MSE")
        print(MSE)

        mape = np.mean(np.abs((demand - forecasts) / demand)) * 100
        print("MAPE")
        print(mape)

        residuals = demand - forecasts

        print("Percentage within 1000 mwh")
        print(sum(list(map(lambda x: int(x <= 1000), abs(residuals))))/len(residuals))

        print("Percentage within 500 mwh")
        print(sum(list(map(lambda x: int(x <= 500), abs(residuals))))/len(residuals))


        plt.plot(results.loc[:, "demand"], label="Demand")
        plt.plot(results.loc[:, "forecast"], label="Forecast")
        plt.legend()
        plt.title("24 hour ahead energy demand forecast for year 2022")
        plt.show()