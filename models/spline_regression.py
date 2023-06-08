import sys
sys.path.insert(0, "C:\\Users\\jerom\\coding\\energy_demand_prediction_framework\\")

import datetime
import numpy as np
import pandas as pd
from patsy import dmatrix
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
from data.quebec_energy_demand import HQ_data


class SplineRegression:

    def get_predictions(self, data, train_start, train_end, test_start, test_end):

        pf = PolynomialFeatures(degree=2)

        # get lagged demand
        lagged_demand = list(data.loc[train_start:test_end - datetime.timedelta(days=1), "demand"])
        new_train_start = train_start + datetime.timedelta(days=1)
        
        data = data.loc[new_train_start:test_end, :]
        data["lagged_demand"] = lagged_demand

        forecasts = []
        for h in range(24):
            hourly_data = data[h::24]

            temp = np.array(hourly_data.loc[:, "temp"])
            basis_x = dmatrix("bs(temp, knots=(15, 22), degree=3, include_intercept=False)", {"temp": temp}, return_type='dataframe')
            basis_x["lags"] = list(hourly_data.loc[:, "lagged_demand"])
            fourier = pd.DataFrame(np.array(self.get_fourier_features(6, 7, hourly_data.loc[:, "day"])))
    
            # train test split
            features = pd.concat([basis_x, fourier], ignore_index=True, axis=1)
            target = np.array(hourly_data.loc[:, "demand"])

            train_features = pf.fit_transform(features[:-1])
            train_target = target[:-1]

            endog = train_target

            model = sm.GLM(endog, train_features).fit()

            # get predictions

            test_features = pf.fit_transform(np.array(features.loc[len(features) -1, :]).reshape(1, -1))

            forecast = model.predict(test_features).tolist()
            forecasts.append(forecast)

        return np.array(forecasts).flatten()



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
    

train_start = datetime.datetime(2021, 1, 1, 0, 0, 0)
train_end = datetime.datetime(2021, 12, 31, 23, 0, 0)
test_start = datetime.datetime(2022, 1, 1, 0, 0,0)
test_end = datetime.datetime(2022, 1, 1, 23, 0, 0)

data = HQ_data().get_history()

sr = SplineRegression()
predictions = sr.get_predictions(data, train_start, train_end, test_start, test_end)
print(predictions)

