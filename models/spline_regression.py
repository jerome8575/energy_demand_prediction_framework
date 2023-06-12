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

        data = data.loc[train_start:test_end, :]
        lag = list(data.loc[:test_end - datetime.timedelta(days=1), "demand"])
        train_start = train_start + datetime.timedelta(days=1)

        augmented_data = data.loc[train_start:, :].copy()
        augmented_data["lag"] = lag

        forecasts = []

        for h in range(24):
            hourly_data = augmented_data[h::24]

            temp = scaler.fit_transform(np.array(hourly_data.loc[:, "temp"]).reshape(-1, 1)).flatten()
            basis_x = dmatrix("bs(temp, knots=(0.65, 0.8), degree=3, include_intercept=False)", {"temp": temp}, return_type='dataframe')
            basis_x["lags"] = np.array(hourly_data.loc[:, "lag"])
            fourier = pd.DataFrame(np.array(self.get_fourier_features(6, 7, hourly_data.loc[:, "day"])))

            exog = pd.concat([basis_x, fourier], ignore_index=True, axis=1)
            target = np.log(np.array(hourly_data.loc[:, "demand"]))

            train_features = exog[:-1]
            train_target = target[:-1]

            model = sm.OLS(train_target, train_features).fit()

            # get predictions
            training_forecasts = model.predict(train_features)
            residuals = training_forecasts - train_target

            res_model = ARIMA(residuals[-50:], order=(1, 0, 1), exog=temp[-51:-1]).fit()
            #training_res_hat = res_model.predict(0, len(residuals[-50:]) - 1, exog=temp[-51:-1])
 
            res_hat = res_model.forecast(1, exog=temp[-1])

            test_features = exog.loc[len(basis_x)-1, :]

            forecast = model.predict(test_features).tolist() - res_hat
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
    


