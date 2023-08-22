
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
from sklearn.linear_model import LinearRegression, Ridge
from sklearn import svm
import xgboost as xgb

pf = PolynomialFeatures(degree=1)

"""spline_regression = pd.read_csv("Ensemble_methods\\individual_predictions\\spline_full.csv")
fourier_regression = pd.read_csv("Ensemble_methods\\individual_predictions\\fourier_full.csv")
mlp = pd.read_csv("Ensemble_methods\\individual_predictions\\mlp_full.csv")"""

spline_regression = pd.read_csv("Ensemble_methods\\individual_predictions\\spline_spain.csv")
fourier_regression = pd.read_csv("Ensemble_methods\\individual_predictions\\fourier_spain.csv")

data = spline_regression.loc[:, ["date_time", "demand", "scaled_temp", "forecast"]].copy()
data["fourier"] = fourier_regression.loc[:, "forecast"]
data["date_time"] = list(map(lambda t: datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S'), data.loc[:, "date_time"]))
data.columns = ["date_time", "demand", "scaled_temp", "spline", "fourier"]
data.set_index("date_time", inplace=True)
print(data)

# add meta-features
mse_window_5 = 5 * 24
mse_window_10 = 10 * 24
window_15 = 15 * 24
data["spline_absolute_error"] = (data.loc[:, "demand"] - data.loc[:, "spline"]).shift(24)
data["spline_mae_10"] = data.loc[:, "spline_absolute_error"].rolling(mse_window_10).mean()
data["fourier_absolute_error"] = (data.loc[:, "demand"] - data.loc[:, "fourier"]).shift(24)
data["fourier_mae_10"] = data.loc[:, "fourier_absolute_error"].rolling(mse_window_10).mean()
data["spline_mae_spline"] = data.loc[:, "spline_mae_10"] * data.loc[:, "spline"]
data["fourier_mae_fourier"] = data.loc[:, "fourier_mae_10"] * data.loc[:, "fourier"]
data["spline_mae_fourier"] = data.loc[:, "spline_mae_10"] * data.loc[:, "fourier"]
data["fourier_mae_spline"] = data.loc[:, "fourier_mae_10"] * data.loc[:, "spline"]


# max error in 15 day window
data["spline_error_max"] = data.loc[:, "spline_absolute_error"].rolling(mse_window_10).max()
data["fourier_error_max"] = data.loc[:, "fourier_absolute_error"].rolling(mse_window_10).max()
data["spline_error_max_spine"] = data.loc[:, "spline_error_max"] * data.loc[:, "spline"]
data["fourier_error_max_fourier"] = data.loc[:, "fourier_error_max"] * data.loc[:, "fourier"]



print(data)
data["constant"] = 1

features = ["spline", "fourier", "spline_mae_10", 
            "fourier_mae_10", "spline_mae_spline", 
            "fourier_mae_fourier", "spline_mae_fourier", 
            "fourier_mae_spline"]
target = "demand"

start_train = datetime.datetime(2021, 1, 1, 0, 0, 0)
end_train = datetime.datetime(2021, 12, 31, 23, 0, 0)
start_test = datetime.datetime(2022, 1, 1, 0, 0, 0)
end_test = datetime.datetime(2022, 12, 31, 23, 0, 0)

print("Individual model performance")
print("Spline regression")
print((mse(data.loc[start_test:end_test, "demand"], data.loc[start_test:end_test, "spline"]))**0.5)
print(mape(data.loc[start_test:end_test, "demand"], data.loc[start_test:end_test, "spline"]))
print("Fourier regression")
print((mse(data.loc[start_test:end_test, "demand"], data.loc[start_test:end_test, "fourier"]))**0.5)
print(mape(data.loc[start_test:end_test, "demand"], data.loc[start_test:end_test, "fourier"]))

model_1 = data.loc[start_test:end_test, "spline"]
model_2 = data.loc[start_test:end_test, "fourier"]
# scale features
scaler = StandardScaler()
data.loc[:, features] = scaler.fit_transform(data.loc[:, features])

# split into train and test

x_train = data.loc[start_train:end_train, features].copy()
y_train = data.loc[start_train:end_train, target].copy() / 10000
x_test = data.loc[start_test:end_test, features].copy()

# MLP neural network

nn = MLPRegressor(max_iter=1000, 
                  activation="identity", solver="adam", learning_rate="adaptive", 
                  learning_rate_init=0.001, alpha=0.0001, batch_size="auto", 
                  verbose=True, early_stopping=True, validation_fraction=0.1, 
                  n_iter_no_change=10, tol=0.0001, random_state=0)
nn.fit(x_train, y_train)

y_pred = nn.predict(x_test) * 10000

results = data.loc[start_test:end_test, ["demand"]].copy()
results["mlp_regressor_forecast"] = y_pred

lreg = LinearRegression()
lreg.fit(x_train, y_train)

y_pred = lreg.predict(x_test) * 10000

results["linear_regression_forecast"] = y_pred

ridge_reg = Ridge(alpha=0.4)
ridge_reg.fit(x_train, y_train)

y_pred = ridge_reg.predict(x_test) * 10000

results["ridge_regression_forecast"] = y_pred

xg_reg = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.01)
xg_reg.fit(x_train, y_train, verbose=True)

y_pred = xg_reg.predict(x_test) * 10000

results["xgboost_forecast"] = y_pred


print("Stacked model performance")

print((mse(results.loc[:, "demand"], results.loc[:, "mlp_regressor_forecast"]))**0.5)
print(mape(results.loc[:, "demand"], results.loc[:, "mlp_regressor_forecast"]))

print((mse(results.loc[:, "demand"], results.loc[:, "linear_regression_forecast"]))**0.5)
print(mape(results.loc[:, "demand"], results.loc[:, "linear_regression_forecast"]))

print((mse(results.loc[:, "demand"], results.loc[:, "ridge_regression_forecast"]))**0.5)
print(mape(results.loc[:, "demand"], results.loc[:, "ridge_regression_forecast"]))

print((mse(results.loc[:, "demand"], results.loc[:, "xgboost_forecast"]))**0.5)
print(mape(results.loc[:, "demand"], results.loc[:, "xgboost_forecast"]))




spline_residual = (results.loc[:, "demand"] - model_1)
fourier_residual = (results.loc[:, "demand"] - model_2)
stacking_residuals = (results.loc[:, "demand"] - results.loc[:, "mlp_regressor_forecast"])

plt.hist(spline_residual, bins=100, label="model 1")
plt.hist(fourier_residual, bins=100, label="model 2")
plt.hist(stacking_residuals, bins=100, label="stacking")
plt.legend()
plt.show()
residuals = abs(results.loc[:, "demand"] - results.loc[:, "mlp_regressor_forecast"])

print("500 MW success")
print(len(residuals[residuals < 500]) / len(residuals))
print("1000 MW success")
print(len(residuals[residuals < 1000]) / len(residuals))
print("1500 MW success")
print(len(residuals[residuals < 1500]) / len(residuals))

plt.plot(results.loc[:, "demand"], label="demand")
plt.plot(results.loc[:, "mlp_regressor_forecast"], label="forecast")
plt.legend()
plt.show()

plt.hist(residuals, bins=100)
plt.show()

results.to_csv("Ensemble_methods\\results\\stacked_model_nn.csv")
