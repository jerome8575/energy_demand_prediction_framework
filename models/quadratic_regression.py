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


class QuadraticRegression:

    def get_predictions(self, data, train_start, train_end, test_start, test_end):

        pf = PolynomialFeatures(degree=2)
        scaler = MinMaxScaler()

        data = data.loc[train_start:test_end, :].copy()

        # get fourier features
        ff_week = self.get_fourier_features(5, 7, data.loc[:, "day"])
        ff_24h = self.get_fourier_features(5, 24, data.loc[:, "hour"])
        fourier_features = pd.concat([ff_week, ff_24h], ignore_index=True, axis=1)

        features = fourier_features
        features["scaled_temp"] = data.loc[:, "scaled_temp"].values
        features["is_clear"] = data.loc[:, "is_clear"].values
        features["temp_15"] = data.loc[:, "temp_15"].values
        features["temp_index_15"] = data.loc[:, "temp_index_15"].values
        features["demand_lag_24"] = data.loc[:, "demand_lag_24"].values
        features["rel_hum"] = data.loc[:, "rel_hum"].values
        features["scaled_temp_diff_24"] = data.loc[:, "scaled_temp_diff_24"].values
        features["scaled_temp_diff_48"] = data.loc[:, "scaled_temp_diff_48"].values

        features["date_time"] = data.index

        features.set_index("date_time", inplace=True)

        target = data.loc[train_start:train_end, "log_demand"].values
        train_features = np.array(pf.fit_transform(np.array(features.loc[train_start:train_end, :].copy())))

        model = sm.OLS(target, train_features).fit()
        train_fit = np.exp(model.predict(train_features))

        residuals = data.loc[train_start:train_end, "demand"] - train_fit
        print(residuals)

        """residual_forecasts = []
        for h in range(24):

            hourly_residual = residuals[residuals.index.hour == h]
            residual_model_start = train_end - datetime.timedelta(days=30)
            residual_model_end = train_end

            residual_data = hourly_residual.loc[residual_model_start:residual_model_end].copy() 

            residual_model = ARIMA(residual_data, order=(1, 0, 1)).fit()
            residual_forecast = residual_model.forecast(1)
            residual_forecasts.append(residual_forecast)"""


        base_forecast = model.predict(np.array(pf.fit_transform(np.array(features.loc[test_start:test_end, :].copy()))))
        forecast  = np.exp(base_forecast) #- np.array(residual_forecasts).flatten()

        return forecast
    

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
