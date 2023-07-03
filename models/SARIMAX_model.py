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


class SARIMAX_model:

    def get_predictions(self, data, train_start, train_end, test_start, test_end):

        pf = PolynomialFeatures(degree=1)

        data = data.loc[train_start:test_end, :].copy()

        forecasts = []
        for h in range(24):

            hourly_data = data[h::24]

            temp = hourly_data.loc[:, ["temp"]]
            temp["temp_squared"] = temp.loc[:, "temp"] ** 2
            ff_week = self.get_fourier_features(6, 7, hourly_data.loc[:, "day"])

            X = pd.concat([temp, ff_week], ignore_index=True, axis=1)

            y_train = np.log(hourly_data.loc[train_start:train_end, "demand"])

            print("y_train: ", y_train.shape)
            x_train = X.loc[train_start:train_end, :]

            print("x_train: ", x_train.shape)
            x_test = X.loc[test_start:test_end, :]

            model = ARIMA(y_train, order=(1, 1, 1), seasonal_order=(1, 0, 1, 7), exog=x_train)
            model_fit = model.fit()

            predictions = model_fit.forecast(len(x_test), exog=x_test)

            forecasts.append(predictions)
        
        return np.exp(np.array(forecasts).flatten())



        """y_train = np.log(data.loc[train_start:train_end, "demand"])

        # get x_train
        ff_week = self.get_fourier_features(6, 7, data.loc[train_start:test_end, "day"])
        ff_day = self.get_fourier_features(6, 24, data.loc[train_start:test_end, "hour"])
        X = pd.concat([ff_week, ff_day], ignore_index=True, axis=1)
        X["temp"] = data.loc[train_start:test_end, "temp"]

        x_train = X.loc[train_start:train_end, :]
        x_test = X.loc[test_start:test_end, :]

        model = ARIMA(y_train, order=(1, 0, 0), seasonal_order=(1, 0, 0, 24), exog=x_train)
        model_fit = model.fit()

        predictions = model_fit.forecast(len(x_test), exog=x_test)
        predictions = np.exp(predictions)

        return predictions"""

    def get_fourier_features(self, n_order, period, values):
        fourier_features = pd.DataFrame(
            {
                f"fourier_{func}_order_{order}_{period}": getattr(np, func)(
                    2 * order * np.pi * values / period
                )
                for order in range(1, n_order + 1)
                for func in ("sin", "cos")
            }
        )
        return fourier_features