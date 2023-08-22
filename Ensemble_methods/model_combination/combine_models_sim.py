
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from statsmodels.tools.eval_measures import rmse
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, StandardScaler
import scipy.optimize as opt
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_squared_error as mse
from sklearn.neural_network import MLPRegressor

from sklearn.linear_model import Ridge
from sklearn.linear_model import LassoCV

pf = PolynomialFeatures(degree=1)

spline_regression = pd.read_csv("Ensemble_methods\\individual_predictions\\spline_full.csv")
fourier_regression = pd.read_csv("Ensemble_methods\\individual_predictions\\fourier_full.csv")
mlp = pd.read_csv("Ensemble_methods\\individual_predictions\\mlp_full.csv")
sarimax = pd.read_csv("Ensemble_methods\\individual_predictions\\sarimax_2019-2022.csv")
bad_forecast = pd.read_csv("Ensemble_methods\\individual_predictions\\forecast_bad.csv")

data = spline_regression.loc[:, ["date_time", "demand", "scaled_temp", "forecast"]].copy()
data["fourier"] = fourier_regression.loc[:, "forecast"]
data["mlp"] = mlp.loc[:, "forecast"]
data["sarimax"] = sarimax.loc[:, "forecast"]
data["date_time"] = list(map(lambda t: datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S'), data.loc[:, "date_time"]))
data.columns = ["date_time", "demand", "scaled_temp", "spline", "fourier", "mlp", "sarimax"]
data.set_index("date_time", inplace=True)
data["fourier_temp"] = data.loc[:, "fourier"] * data.loc[:, "scaled_temp"]
data["mlp_temp"] = data.loc[:, "mlp"] * data.loc[:, "scaled_temp"]
data["spline_temp"] = data.loc[:, "spline"] * data.loc[:, "scaled_temp"]

mse_window = 10*24
data["spline_squared_error"] = (data.loc[:, "demand"] - data.loc[:, "spline"]).shift(24)
data["spline_mse"] = data.loc[:, "spline_squared_error"].rolling(mse_window).mean()
data["fourier_squared_error"] = (data.loc[:, "demand"] - data.loc[:, "fourier"]).shift(24)
data["fourier_mse"] = data.loc[:, "fourier_squared_error"].rolling(mse_window).mean()
data["mlp_squared_error"] = (data.loc[:, "demand"] - data.loc[:, "mlp"])
data["mlp_mse"] = data.loc[:, "mlp_squared_error"].rolling(mse_window).mean()

data["spline_mse_spline"] = data.loc[:, "spline_mse"] * data.loc[:, "spline"]
data["fourier_mse_fourier"] = data.loc[:, "fourier_mse"] * data.loc[:, "fourier"]
data["mlp_mse_mlp"] = data.loc[:, "mlp_mse"] * data.loc[:, "mlp"]

data["spline_mse_fourier"] = data.loc[:, "spline_mse"] * data.loc[:, "fourier"]
data["fourier_mse_spline"] = data.loc[:, "fourier_mse"] * data.loc[:, "spline"]
data["constant"] = 1

print(data)


# Simulation

# baseline neural network

"""scaler = StandardScaler()
features = ["spline", "fourier", "mlp", "spline_mse", "fourier_mse", "mlp_mse"]
target = "demand"

start_train = datetime.datetime(2020, 11, 1, 0, 0, 0)
end_train = datetime.datetime(2020, 12, 31, 23, 0, 0)
start_test = datetime.datetime(2021, 1, 1, 0, 0, 0)
end_test = datetime.datetime(2021, 1, 1, 23, 0, 0)

scaled_features = scaler.fit_transform(data.loc[:, features])
data.loc[:, features] = scaled_features

forecasts = []
for i in range(365):
    print(i)
    train = data.loc[start_train:end_train, :].copy()
    test = data.loc[start_test:end_test, :].copy()

    nn = MLPRegressor(hidden_layer_sizes=(100, 100, 100), activation="relu", solver="adam", max_iter=1000, random_state=42)
    nn.fit(train.loc[:, features], train.loc[:, target]/10000)

    forecast = nn.predict(test.loc[:, features]) * 10000
    forecasts.append(forecast)

    start_train += datetime.timedelta(days=1)
    end_train += datetime.timedelta(days=1)
    start_test += datetime.timedelta(days=1)
    end_test += datetime.timedelta(days=1)

forecasts = np.array(forecasts).flatten()

test = data.loc[datetime.datetime(2021, 1, 1, 0, 0, 0):datetime.datetime(2021, 1, 1, 0, 0, 0) + datetime.timedelta(days=365) - datetime.timedelta(hours=1), ["demand"]].copy()
test["forecast"] = forecasts

print((mse(test.loc[:, "demand"], test.loc[:, "forecast"]))**0.5)
print(mape(test.loc[:, "demand"], test.loc[:, "forecast"]))"""


# linear regression approach
scaler = StandardScaler()
demand_scaler = StandardScaler()
features = ["spline", "fourier", "mlp", "sarimax"]
target = "demand"

window_size = 30

start_sim = datetime.datetime(2021, 1, 1, 0, 0, 0)
start_train = start_sim - datetime.timedelta(days=window_size)
start_train_validation = start_train
end_train = start_sim - datetime.timedelta(days=1)
end_train_validation = start_sim - datetime.timedelta(days=2)

forecasts = []
ols_forecasts = []
parameters = []
ols_parameters = []
alpha_range = np.arange(0.001, 0.1, 0.001)
ridge_rmses = []
ols_rmses = []
alpha_opts = []
for i in range(365 * 24):
    print(i)
    data_iter = data.loc[start_train:start_sim, :].copy()
    data_iter_features = scaler.fit_transform(data_iter.loc[:, features])
    data_iter.loc[:, features] = data_iter_features

    non_scaled_x_train = pf.fit_transform(data.loc[start_train:end_train, features].copy())
    non_scaled_y_train = data.loc[start_train:end_train, target].copy()
    non_scaled_x_test = pf.fit_transform(np.array(data.loc[start_sim, features].copy()).reshape(1, -1))
    x_train = pf.fit_transform(data_iter.loc[start_train:end_train, features].copy())
    y_train = demand_scaler.fit_transform(np.array(data.loc[start_train:end_train, target].copy()).reshape(-1, 1))
    x_test = pf.fit_transform(np.array(data_iter.loc[start_sim, features].copy()).reshape(1, -1))
    x_train_validate = pf.fit_transform(data_iter.loc[start_train:end_train_validation, features].copy())
    y_train_validate = demand_scaler.fit_transform(np.array(data.loc[start_train:end_train_validation, target].copy()).reshape(-1, 1))
    x_validate = pf.fit_transform(np.array(data_iter.loc[end_train, features].copy()).reshape(1, -1))
    y_validate = data_iter.loc[end_train, target]
    y_test = data_iter.loc[start_sim, target]
    rmse_min = 100000000
    alpha_opt = 0
    """for alpha in alpha_range:

        model = sm.OLS(y_train, x_train).fit_regularized(alpha=alpha, L1_wt=0)
        forecast = demand_scaler.inverse_transform(model.predict(x_test).reshape(-1, 1))
        rmse = (forecast - y_test)**2

        if rmse < rmse_min:
            rmse_min = rmse
            alpha_opt = alpha"""

    #alpha_opts.append(alpha_opt)


    model = sm.OLS(y_train, x_train).fit_regularized(alpha=0.05, L1_wt=0)
    forecast_ridge = model.predict(x_test)
    forecast_ridge = demand_scaler.inverse_transform(forecast_ridge.reshape(-1, 1))
    """ridgecv_model = Ridge(alpha=30, positive=True, fit_intercept=False).fit(x_train, y_train)
    ridgecv_forecast = ridgecv_model.predict(x_test)
    ridgecv_forecast = demand_scaler.inverse_transform(ridgecv_forecast.reshape(-1, 1))"""

    ols_model = sm.OLS(y_train, x_train).fit()
    forecast_ols = demand_scaler.inverse_transform(ols_model.predict(x_test).reshape(-1, 1))

    # ridgecv

    #normal equations
    x_train = np.array(non_scaled_x_train)
    y_train = np.array(non_scaled_y_train)
    x_test = np.array(non_scaled_x_test)
    
    params = np.linalg.inv(x_train.T @ x_train) @ x_train.T @ y_train

    normal_eq_forecast = x_test @ params


    forecast = forecast_ridge

    #print(ols_model.params)
    parameters.append(model.params)
    ols_parameters.append(ols_model.params)
    forecasts.append(forecast[0])
    ols_forecasts.append(forecast_ols[0])

    start_sim += datetime.timedelta(hours=1)
    start_train += datetime.timedelta(hours=1)
    end_train += datetime.timedelta(hours=1)
    end_train_validation += datetime.timedelta(hours=1)


"""ridge_rmses = np.array(ridge_rmses).flatten()
ols_rmses = np.array(ols_rmses).flatten()

print(ridge_rmses)
print(ols_rmses)

plt.plot(ols_rmses, color="blue", label="ols")
plt.plot(ridge_rmses, color="red", label="ridge")
plt.legend()
plt.show()"""

results = data.loc[datetime.datetime(2021, 1, 1, 0, 0, 0): datetime.datetime(2021, 12, 31, 23, 0, 0), ["demand", "spline", "fourier", "mlp", "sarimax"]].copy()
results["forecast"] = np.array(forecasts).flatten()
results["ols_forecasts"] = np.array(ols_forecasts).flatten()
# results["ridge_forecasts"] = np.array(en_forecasts).flatten()
print(forecasts)
parameters = pd.DataFrame(parameters, columns=["intercept", "spline", "fourier", "mlp", "sarimax"])
ols_parameters = pd.DataFrame(ols_parameters, columns=["intercept", "spline", "fourier", "mlp", "sarimax"])
print(results)
print(parameters)

"""results["alpha_opt"] = alpha_opts
plt.hist(results.loc[:, "alpha_opt"])
plt.show()
print("Mean alpha opt")
print(np.mean(results.loc[:, "alpha_opt"]))"""

"""hour = 15
alpha_opt_hour = results.loc[:, "alpha_opt"].values[hour::15]
plt.plot(alpha_opt_hour)
plt.show()"""

plt.plot(ols_parameters.loc[:, "spline"], color="red", label="spline")
plt.plot(ols_parameters.loc[:, "fourier"], color="blue", label="fourier")
#plt.plot(ols_parameters.loc[:, "fourier"], color="black", label="ols")
plt.plot(ols_parameters.loc[:, "mlp"], color="green", label="mlp")
plt.plot(ols_parameters.loc[:, "sarimax"], color="orange", label="sarimax")
plt.xlabel("Time")
plt.ylabel("Model weight")
plt.legend()
plt.show()


print((mse(results.loc[:, "demand"], results.loc[:, "forecast"]))**0.5)
print(mape(results.loc[:, "demand"], results.loc[:, "forecast"]))

print((mse(results.loc[:, "demand"], results.loc[:, "ols_forecasts"]))**0.5)
print(mape(results.loc[:, "demand"], results.loc[:, "ols_forecasts"]))

print((mse(results.loc[:, "demand"], results.loc[:, "fourier"]))**0.5)
print(mape(results.loc[:, "demand"], results.loc[:, "fourier"]))



residuals = abs(np.array(results.loc[:, "demand"] - results.loc[:, "forecast"]))
spline_residuals = abs(np.array(results.loc[:, "demand"] - results.loc[:, "spline"]))
fourier_residuals = abs(np.array(results.loc[:, "demand"] - results.loc[:, "fourier"]))
mlp_residuals = abs(np.array(results.loc[:, "demand"] - results.loc[:, "mlp"]))
sarimax_residuals = abs(np.array(results.loc[:, "demand"] - results.loc[:, "sarimax"]))

results["residuals"] = residuals
results["spline_res"] = spline_residuals
results["fourier_res"] = fourier_residuals
results["mlp_res"] = mlp_residuals
results["sarimax_res"] = sarimax_residuals

print(residuals)

print("500 MW success")
print(len(residuals[residuals < 500]) / len(residuals))
print("1000 MW success")
print(len(residuals[residuals < 1000]) / len(residuals))
print("1500 MW success")
print(len(residuals[residuals < 1500]) / len(residuals))

plt.hist(results.loc[:, ["spline_res", "residuals"]], bins=100, label=["Best model", "Ridge ensemble"])
"""plt.hist(results.loc[:, "fourier_res"], bins=100, color="black")
plt.hist(results.loc[:, "mlp_res"], bins=100, color="black")
plt.hist(results.loc[:, "sarimax_res"], bins=100, color="black")"""
#plt.hist(results.loc[:, "residuals"], bins=100, color="orange")
plt.legend()
plt.title("Distribution of forecasting errors")
plt.show()

# plot results


#parameters.to_csv("Ensemble_methods\\results\\stacked_model_ols_parameters.csv")
 

# static linear regression approach

"""features = ["spline", "fourier", "spline_mse", "fourier_mse", "spline_mse_spline", "fourier_mse_fourier", "spline_mse_fourier", "fourier_mse_spline"]
target = "demand"

train_start = datetime.datetime(2020, 1, 1, 0, 0, 0)
train_end = datetime.datetime(2020, 12, 31, 23, 0, 0)
test_start = datetime.datetime(2021, 1, 1, 0, 0, 0)
test_end = datetime.datetime(2021, 12, 31, 23, 0, 0)


x_train = pf.fit_transform(data.loc[train_start:train_end, features].copy())
y_train = data.loc[train_start:train_end, target].copy()
x_test = pf.fit_transform(np.array(data.loc[test_start:test_end, features].copy()))

model = sm.OLS(y_train, x_train).fit()
forecast = model.predict(x_test)
print(model.summary())

results = data.loc[test_start:test_end, ["demand"]].copy()
results["forecast"] = forecast

print(results)

print((mse(results.loc[:, "demand"], results.loc[:, "forecast"]))**0.5)
print(mape(results.loc[:, "demand"], results.loc[:, "forecast"]))

plt.plot(results.loc[:, "demand"], label="demand")
plt.plot(results.loc[:, "forecast"], label="forecast")
plt.legend()
plt.show()

residuals = abs(results.loc[:, "demand"] - results.loc[:, "forecast"])

plt.hist(residuals, bins=100)
plt.show()

print("500 MW success rate")
print(len(residuals[residuals <= 500]) / len(residuals))
print("1000 MW success rate")
print(len(residuals[residuals <= 1000]) / len(residuals))
print("1500 MW success rate")
print(len(residuals[residuals <= 1500]) / len(residuals))
"""
