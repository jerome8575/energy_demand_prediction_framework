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


class SplineRegression:

    def get_predictions(self, data, train_start, train_end, test_start, test_end):

        pf = PolynomialFeatures(degree=1)
        scaler = MinMaxScaler()

        data = data.loc[train_start:test_end, :].copy()

        temp = scaler.fit_transform(np.array(data.loc[:, "temp"]).reshape(-1, 1)).flatten()
        demand = np.log(data.loc[:, "demand"])

        data["log_demand"] = demand
        data["scaled_temp"] = temp

        forecasts = []
        start_plus_one_day = train_start + datetime.timedelta(days=2)
        for h in range(24):
            """hourly_data = augmented_data[h::24]

            temp = scaler.fit_transform(np.array(hourly_data.loc[:, "temp"]).reshape(-1, 1)).flatten()
            basis_x = dmatrix("bs(temp, knots=(0.4, 0.6), degree=3, include_intercept=False)", {"temp": temp}, return_type='dataframe')
            basis_x["lags"] = np.array(hourly_data.loc[:, "lag"])
            basis_x["is_cloudy"] = np.array(hourly_data.loc[:, "is_cloudy"])
            basis_x["is_clear"] = np.array(hourly_data.loc[:, "is_clear"])
            basis_x["is_snowing"] = np.array(hourly_data.loc[:, "is_snowing"])"""

            hourly_data = data[h::24]

            temp_diff = np.diff(hourly_data.loc[:, "scaled_temp"])
            demand_lag = np.array(hourly_data.loc[:, "log_demand"])[:-1]
            demand_lag_lag = np.array(hourly_data.loc[:, "log_demand"])[:-2]

            temp = np.array(hourly_data.loc[start_plus_one_day:, "scaled_temp"])
            basis_x = dmatrix("bs(scaled_temp, knots=(0.4, 0.6), degree=3, include_intercept=False)", {"scaled_temp": temp}, return_type='dataframe')

            hourly_data = hourly_data.loc[start_plus_one_day:, :].copy()

            basis_x["temp_diff"] = temp_diff[1:]
            basis_x["temp_squared"] = temp ** 2
            basis_x["lags"] = demand_lag[1:]
            basis_x["lags_lag"] = demand_lag_lag
            basis_x["is_clear"] = list(hourly_data.loc[:, "is_clear"])

            fourier = pd.DataFrame(np.array(self.get_fourier_features(6, 7, hourly_data.loc[:, "day"])))

            exog = pd.concat([basis_x, fourier], ignore_index=True, axis=1)
            target = np.array(hourly_data.loc[:, "log_demand"])

            train_features = exog[:-1]
            train_target = target[:-1]

            model = sm.OLS(train_target, train_features).fit()

            # get predictions
            """training_forecasts = model.predict(train_features)
            residuals = training_forecasts - train_target

            res_model = ARIMA(residuals[-50:], order=(1, 0, 1), exog=temp[-51:-1]).fit()
            #training_res_hat = res_model.predict(0, len(residuals[-50:]) - 1, exog=temp[-51:-1])
 
            res_hat = res_model.forecast(1, exog=temp[-1])"""

            test_features = exog.loc[len(basis_x)-1, :]

            forecast = model.predict(test_features).tolist() #- res_hat
            forecasts.append(forecast)


        return np.exp(np.array(forecasts).flatten())



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
    


