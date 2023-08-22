import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from statsmodels.tools.eval_measures import rmse
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
import scipy.optimize as opt
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.linear_model import RidgeCV

pf = PolynomialFeatures(degree=1)


# All models we have
spline_regression = pd.read_csv("Ensemble_methods\\individual_predictions\\spline_full.csv")
quadratic_regression = pd.read_csv("Ensemble_methods\\individual_predictions\\fourier_full.csv")
"""bayes_forecast = pd.read_csv("Ensemble_methods\\individual_predictions\\bayes_forecasts_1.csv")
sarimax = pd.read_csv("Ensemble_methods\\individual_predictions\\sarimax_2021_cut.csv")
mlp = pd.read_csv("Ensemble_methods\\individual_predictions\\mlp_forecasts.csv")
mlp_2 = pd.read_csv("Ensemble_methods\\individual_predictions\\mlp_forecasts_2.csv")"""

data = pd.concat([spline_regression.loc[:, ["date_time", "demand", "scaled_temp", "forecast"]], 
                  quadratic_regression.loc[:, "forecast"]],
                  ignore_index=True, axis=1)

data.columns = ["date_time", "demand", "scaled_temp", "spline", "quadratic"]
data["date_time"] = list(map(lambda t: datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S'), data.loc[:, "date_time"]))
data.set_index("date_time", inplace=True)

start = datetime.datetime(2021, 1, 1, 0, 0, 0)
end = datetime.datetime(2021, 12, 31, 23, 0, 0)

data = data.loc[start:end, :].copy()


# Simulation
# choose window size for regression
window_size_days = 30
# choose models to combine
models = ["spline", "quadratic"]

forecasts = pd.DataFrame()
parameters = []
for hour in range(24):
    data_hour = data[data.index.hour == hour].copy()
    sim_start = start
    sim_current = sim_start + datetime.timedelta(days=window_size_days) + datetime.timedelta(hours=hour)
    sim_end_train = sim_current - datetime.timedelta(days=1)
    for i in range((365 - window_size_days)):
        # calculate loss
        data_window = data_hour.loc[sim_start:sim_current, :].copy()
        print(data_window)
        # regression approach
        model = sm.OLS(data_window.loc[:sim_end_train, "demand"], pf.fit_transform(data_window.loc[:sim_end_train, models])).fit_regularized(alpha=0.04)
        parameters.append(model.params)
        forecast_regression = model.predict(pf.fit_transform(np.array(data_window.loc[sim_current, models]).reshape(1, -1)))
        forecasts.loc[sim_current.date(), "hour" + str(hour)] = forecast_regression[0]
        sim_start = sim_start + datetime.timedelta(days=1)
        sim_current = sim_current + datetime.timedelta(days=1)
        sim_end_train = sim_end_train + datetime.timedelta(days=1)


print(forecasts)
forecasts = np.array(forecasts).flatten()
parameters = pd.DataFrame(parameters, columns=["intercept", "spline", "quadratic"])
plt.plot(parameters.loc[:, "spline"])
plt.plot(parameters.loc[:, "quadratic"])
plt.show()

results = data.loc[start + datetime.timedelta(days=window_size_days):end, ["demand"]].copy()
results["combined_forecast"] =  forecasts

print("MAPE")
print(mape(results.loc[:, "demand"], results.loc[:, "combined_forecast"]))
print("RMSE")
print(rmse(results.loc[:, "demand"], results.loc[:, "combined_forecast"]))

residuals = abs(results.loc[:, "demand"] - results.loc[:, "combined_forecast"])

print("1500 MW success")
print(len(residuals[residuals < 1500]) / len(residuals))
print("1000 MW success")
print(len(residuals[residuals < 1000]) / len(residuals))
print("500 MW success")
print(len(residuals[residuals < 500]) / len(residuals))


print(results)