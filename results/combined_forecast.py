
# get two forecasts as np.array
# weighted sum
# plot

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from statsmodels.tools.eval_measures import rmse
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
import scipy.optimize as opt
from sklearn.metrics import mean_absolute_percentage_error as mape

pf = PolynomialFeatures(degree=1)

spline_regression = pd.read_csv("results\spline_regression_2021.csv")
quadratic_regression = pd.read_csv("results\quadratic_fourier_regression_2021.csv")
l_inf_norm_min = pd.read_csv("results\inf_norm_reg_2021.csv")
sarimax = pd.read_csv("results\sarimax_2021.csv")

quad_spain = pd.read_csv("results\quad_spain_2021.csv")
spline_spain = pd.read_csv("results\spline_spain_2021.csv")
spline_corrected_spain = pd.read_csv("results\spline_spain_corrected_2021.csv")

data = pd.concat([spline_spain.loc[:, ["date_time", "demand", "scaled_temp", "forecast"]], quad_spain.loc[:, "forecast"], spline_corrected_spain.loc[:, "forecast"]], ignore_index=True, axis=1)
data.columns = ["date_time", "demand", "scaled_temp", "spline", "quadratic", "spline_corrected"]
data["date_time"] = list(map(lambda t: datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S'), data.loc[:, "date_time"]))
data.set_index("date_time", inplace=True)

print(data)

start = datetime.datetime(2021, 1, 1, 0, 0, 0)
end = datetime.datetime(2021, 12, 30, 23, 0, 0)

data = data.loc[start:end, :].copy()


target = np.array(data.loc[:, "demand"])
exog = pf.fit_transform(data.loc[:, ["spline", "quadratic", "spline_corrected"]])

def linf_norm_min(params, covariates, target):
        vector = np.abs(target - params.dot(covariates).transpose()) 
        return np.max(vector)

def l1_norm_min(params, covariates, target):
    vector = np.abs(target - params.dot(covariates).transpose()) 
    return np.sum(vector)

model = sm.OLS(target, exog).fit()
start_params = np.array(model.params)
covariates = np.array(exog).transpose()

result_inf = opt.minimize(linf_norm_min, x0=start_params, args=(covariates, target), method="BFGS")
result_l1 = opt.minimize(l1_norm_min, x0=start_params, args=(covariates, target), method="BFGS")

# combine model
#combined_forecast_benchmark = 0.5 * data.loc[:, "spline"] + 0.5 * data.loc[:, "quadratic"]
combined_forecast_regression = model.predict(exog).tolist()
combined_forecast_inf = result_inf.x.dot(covariates).transpose()
combined_forecast_l1 = result_l1.x.dot(covariates).transpose()


results = data.loc[:, ["demand"]]
#results["forecast_benchmark"] = combined_forecast_benchmark
results["forecast_reg"] = combined_forecast_regression
results["forecast_linf"] = combined_forecast_inf
results["forecast_l1"] = combined_forecast_l1

print("RMSE for regression forecast: ")
print(rmse(np.array(results.loc[:, "demand"]), np.array(results.loc[:, "forecast_reg"])))

print("RMSE for linf forecast: ")
print(rmse(np.array(results.loc[:, "demand"]), np.array(results.loc[:, "forecast_linf"])))

print("RMSE for l1 forecast: ")
print(rmse(np.array(results.loc[:, "demand"]), np.array(results.loc[:, "forecast_l1"])))

#print("RMSE for benchmark forecast: ")
#print(rmse(np.array(results.loc[:, "demand"]), np.array(results.loc[:, "forecast_benchmark"])))

print("MAPE for regression forecast: ")
print(mape(np.array(results.loc[:, "demand"]), np.array(results.loc[:, "forecast_reg"])))

linf_residuals = np.abs(np.array(results.loc[:, "demand"]) - np.array(results.loc[:, "forecast_linf"]))
l1_residuals = np.abs(np.array(results.loc[:, "demand"]) - np.array(results.loc[:, "forecast_l1"]))
l2_residuals = np.abs(np.array(results.loc[:, "demand"]) - np.array(results.loc[:, "forecast_reg"]))
#benchmark_residuals = np.abs(np.array(results.loc[:, "demand"]) - np.array(results.loc[:, "forecast_benchmark"]))


# 1000 megawatt success
print("1000 megawatt success: ")
print("linf: ")
print(len(linf_residuals[linf_residuals <= 1000]) / len(linf_residuals))
print("l1: ")
print(len(l1_residuals[l1_residuals <= 1000]) / len(l1_residuals))
print("l2: ")
print(len(l2_residuals[l2_residuals <= 1000]) / len(l2_residuals))
#print("benchmark: ")
#print(len(benchmark_residuals[benchmark_residuals <= 1000]) / len(benchmark_residuals))

plt.hist(linf_residuals, bins=100, label="linf")
plt.hist(l1_residuals, bins=100, label="l1")
plt.hist(l2_residuals, bins=100, label="l2")
#plt.hist(benchmark_residuals, bins=100, label="benchmark")
plt.legend()
plt.show()

plt.plot(results.loc[:, ["demand", "forecast_reg", "forecast_linf", "forecast_l1"]], label=["demand", "forecast_reg", "forecast_linf", "forecast_l1"])
plt.legend()
plt.show()

results.to_csv("results\combined_forecast.csv")




