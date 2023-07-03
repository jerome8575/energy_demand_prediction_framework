import sys
sys.path.insert(0, "C:\\Users\\jerom\\coding\\energy_demand_prediction_framework\\")

import datetime
import numpy as np
import pandas as pd
from patsy import dmatrix
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
import statsmodels.api as sm
from data.quebec_energy_demand import HQ_data
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt


class STRregression:

    def get_predictions(self, data, train_start, train_end, test_start, test_end):

        pf = PolynomialFeatures(degree=1)
        scaler = MinMaxScaler()

        data = data.loc[train_start:test_end, :].copy()

        temp = scaler.fit_transform(np.array(data.loc[:, "temp"]).reshape(-1, 1)).flatten()
        demand = np.log(data.loc[:, "demand"])

        data["log_demand"] = demand
        data["scaled_temp"] = temp

        forecasts = []
        start_plus_one_day = train_start + datetime.timedelta(days=1)
        for h in range(24):

            hourly_data = data[h::24]

            temp_diff = np.diff(hourly_data.loc[:, "temp"])
            demand_lag = np.array(hourly_data.loc[:, "log_demand"])[:-1]

            hourly_data = hourly_data.loc[start_plus_one_day:, :].copy()

            features = hourly_data.loc[:, ["temp", "is_weekend"]]

            features["temp_diff"] = temp_diff
            features["lags"] = demand_lag

            features = features.reset_index(drop=True)
            target = np.array(hourly_data.loc[:, "log_demand"])

            train_features = pf.fit_transform(features[:-1])
            train_target = target[:-1]

            model = sm.OLS(train_target, train_features).fit()

            test_features = pf.fit_transform(np.array(features.loc[len(features)-1, :]).reshape(1, -1))

            forecast = model.predict(test_features).tolist()
            forecasts.append(forecast)


        return np.exp(np.array(forecasts).flatten())
