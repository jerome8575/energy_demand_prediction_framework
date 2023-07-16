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
import scipy.optimize as opt


class InfiniteNormMinimization:

    def get_predictions(self, data, train_start, train_end, test_start, test_end):

        pf = PolynomialFeatures(degree=1)
        scaler = MinMaxScaler()

        data = data.loc[train_start:test_end, :].copy()

        temp = np.array(data.loc[:, "scaled_temp"])

        forecasts = {}
        base_forecasts = []
        max_corr_lags_winter = [7, 7, 7, 8, 9, 8, 9, 10, 11, 11, 12, 13, 14, 15, 1, 1, 1, 2, 2, 3, 4, 6, 6, 7]
        for h in range(24):

            hourly_data = data[h::24]

            temp = np.array(hourly_data.loc[:, "scaled_temp"])
            basis_x = dmatrix("bs(scaled_temp, knots=(0, 0.5, 1, 1.3), degree=1, include_intercept=True)", {"scaled_temp": temp}, return_type='dataframe')

            basis_x["demand_lag"] = np.array(hourly_data.loc[:, "demand_lag_24"])
            basis_x["is_clear"] = np.array(hourly_data.loc[:, "is_clear"])
            basis_x["temp_1"] = np.array(hourly_data.loc[:, "temp_lag_1"]) 
            basis_x["temp_15"] = np.array(hourly_data.loc[:, "temp_lag_15"])
            basis_x["is_weekend"] = np.array(hourly_data.loc[:, "is_weekend"])
            #basis_x["wind_speed"] = np.array(hourly_data.loc[:, "wind_speed"])
            basis_x["rel_hum"] = np.array(hourly_data.loc[:, "rel_hum"])
            basis_x["rel_hum_temp"] = np.array(hourly_data.loc[:, "rel_hum"]) * np.array(hourly_data.loc[:, "scaled_temp"]) 
            basis_x["temp_high_corr"] = np.array(hourly_data.loc[:, "temp_lag_"+str(max_corr_lags_winter[h])])
            basis_x["temp_high_corr_inde"] = np.array(hourly_data.loc[:, "temp_index_"+str(max_corr_lags_winter[h])])

            basis_x["date_time"] = hourly_data.index
            basis_x.set_index("date_time", inplace=True)

            exog = basis_x
            target = hourly_data.loc[:, "log_demand"]

            train_features = exog.loc[train_start:train_end, :]
            train_target = target.loc[train_start:train_end]

            model = sm.OLS(train_target, train_features).fit()

            start_params = np.array(model.params)
            covariates = np.array(train_features).transpose()

            result = opt.minimize(self.l_inf_norm_min, x0=start_params, args=(covariates, train_target), method="Nelder-Mead")

            test_features = exog.loc[test_start:test_end, :]
            test_covariates = np.array(test_features).transpose()
            base_forecast = result.x.dot(test_covariates).transpose()

            forecast = base_forecast
            #print(len(forecast)) #- res_hat
            forecasts["h_" + str(h)] = forecast

        # retrieve continuous forecasts

        forecasts = pd.DataFrame(forecasts)

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
    
    def l_inf_norm_min(self, params, covariates, target):
        vector = np.abs(target - params.dot(covariates).transpose()) 
        return np.max(vector)
    


