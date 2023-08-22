
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
from statsmodels.stats.outliers_influence import variance_inflation_factor

pf = PolynomialFeatures(degree=1)
scaler = StandardScaler()
demand_scaler = StandardScaler()

spline_regression = pd.read_csv("Ensemble_methods\\individual_predictions\\spline_full.csv")
fourier_regression = pd.read_csv("Ensemble_methods\\individual_predictions\\fourier_full.csv")
mlp = pd.read_csv("Ensemble_methods\\individual_predictions\\mlp_full.csv")
sarimax = pd.read_csv("Ensemble_methods\\individual_predictions\\sarimax_2019-2022.csv")

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

features = ["spline", "fourier", "sarimax", "mlp"]
target = "demand"

"""print(data.loc[:, ["spline", "fourier", "mlp", "sarimax"]].corr())
print(data.loc[:, ["spline", "fourier", "mlp", "sarimax"]].cov())"""

"""corr_mat = np.dot(np.array(data.loc[:, ["constant", "spline", "fourier", "mlp", "sarimax"]]).transpose(), np.array(data.loc[:, ["constant", "spline", "fourier", "mlp", "sarimax"]]))
print(corr_mat)"""

X = scaler.fit_transform(data.loc[:, ["spline", "fourier", "mlp", "sarimax"]].copy())
Xt = X.transpose()

corr_mat = np.dot(Xt, X) / len(X)
print(corr_mat)
print("eigen values")
print(np.linalg.eigvals(corr_mat))
#window_size = 30

start_train = datetime.datetime(2021, 1, 1, 0, 0, 0)
end_train = datetime.datetime(2021, 3, 31, 23, 0, 0)
start_sim = datetime.datetime(2021, 4, 1, 0, 0, 0)
end_sim = datetime.datetime(2021, 4, 1, 23, 0, 0)

scaled_features = scaler.fit_transform(data.loc[:, features].copy())
scaled_target = demand_scaler.fit_transform(np.array(data.loc[:, target].copy()).reshape(-1, 1))
data[features] = scaled_features
data["scaled_demand"] = scaled_target

x_train = np.array(data.loc[start_train:end_train, features].copy())
y_train = np.array(data.loc[start_train:end_train, "scaled_demand"].copy())
x_test = np.array(data.loc[start_sim:end_sim, features].copy())

"""plt.scatter(data.loc[:, "spline"], data.loc[:, "fourier"])
plt.xlabel("Spline regression")
plt.ylabel("Fourier series")
plt.title("Spline regression model forecast vs Fourier series model forecast")
plt.show()"""

# calculate VIF
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(x_train, i) for i in range(x_train.shape[1])]
print(vif)


print("COllinearity")
print(pd.DataFrame(x_train).loc[:, :].corr())

y_test = data.loc[start_sim:end_sim, "demand"].copy()
rmses = []
alpha_range = np.arange(0, 0.3, 0.001)
parameters = pd.DataFrame(columns=["spline", "fourier", "mlp", "sarimax"])
for alpha in alpha_range:

    model = sm.OLS(y_train, x_train).fit_regularized(alpha=alpha, L1_wt=0, refit=True)
    forecasts = demand_scaler.inverse_transform(model.predict(x_test).reshape(-1,1))
    rmse = (mse(y_test, forecasts))**0.5
    #print(rmse)
    rmses.append(rmse)
    parameters.loc[alpha, :] = model.params


plt.plot(parameters.loc[:, "spline"], label="spline")
plt.plot(parameters.loc[:, "fourier"], label="fourier")
plt.plot(parameters.loc[:, "mlp"], label="mlp")
plt.plot(parameters.loc[:, "sarimax"], label="sarimax")
plt.xlabel("Lambda")
plt.ylabel("Parameter value")
plt.title("Parameter values vs penalty parameter lambda")
plt.legend()
plt.show()

ols_model = sm.OLS(y_train, x_train).fit()
print(ols_model.summary())
print(ols_model.mse_model)
ols_forecasts = demand_scaler.inverse_transform(ols_model.predict(x_test).reshape(-1, 1))
ols_rmse = mse(y_test, ols_forecasts)**0.5
print("OLS")
print(ols_rmse)
print(mape(y_test, ols_forecasts))

#normal equation solution
normal_equation = np.dot(np.dot(np.linalg.inv(np.dot(x_train.transpose(), x_train)), x_train.transpose()), y_train)
normal_equation_forecasts = demand_scaler.inverse_transform(np.dot(x_test, normal_equation).reshape(-1, 1))
normal_equation_rmse = mse(y_test, normal_equation_forecasts)**0.5
print("Normal equation")
print(normal_equation_rmse)
print(mape(y_test, normal_equation_forecasts))

ridge_model = sm.OLS(y_train, x_train).fit_regularized(alpha=0.1, L1_wt=0, refit=True)
ridge_forecasts = demand_scaler.inverse_transform(ridge_model.predict(x_test).reshape(-1, 1))
ridge_rmse = mse(y_test, ridge_forecasts)**0.5
print("Ridge")
print(ridge_rmse)
print(mape(y_test, ridge_forecasts))



plt.plot(alpha_range, rmses, color="blue", label="regularized LS RMSE")
plt.xlabel("Lambda")
plt.ylabel("RMSE")
plt.title("Validation RMSE vs penalty parameter lambda")
plt.plot(alpha_range, [ols_rmse] * len(alpha_range), linestyle="--", color="red", label="OLS RMSE")
plt.legend()
plt.show()

"""results = data.loc[start_sim: end_sim, ["demand"]].copy()
results["forecast"] = forecasts

print((mse(results.loc[:, "demand"], results.loc[:, "forecast"]))**0.5)
print(mape(results.loc[:, "demand"], results.loc[:, "forecast"]))"""