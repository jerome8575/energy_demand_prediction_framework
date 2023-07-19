
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
quad_quebec_temp = pd.read_csv("results\\fourier_quebec_weather.csv")

"""plt.plot(spline_regression.loc[:, "demand"], label="Demand")
plt.plot(quad_quebec_temp.loc[:, "forecast"], label="quebec temp")
plt.plot(quadratic_regression.loc[:, "forecast"], label="Fourier series model")
plt.legend()
plt.show()

spline_residuals = spline_regression.loc[:, "demand"] - spline_regression.loc[:, "forecast"]
quadratic_residuals = quadratic_regression.loc[:, "demand"] - quadratic_regression.loc[:, "forecast"]

plt.hist(spline_residuals, bins=100, label="Regression splines model")
plt.hist(quadratic_residuals, bins=100, label="Fourier series model")
plt.legend()
plt.show()"""

# choose model to combine
data = pd.concat([spline_corrected_spain.loc[:, ["date_time", "demand", "scaled_temp", "forecast"]], 
                  quad_spain.loc[:, "forecast"]],
                  ignore_index=True, axis=1)

data.columns = ["date_time", "demand", "scaled_temp", "spline", "quadratic"]
data["date_time"] = list(map(lambda t: datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S'), data.loc[:, "date_time"]))
data.set_index("date_time", inplace=True)

print(data)

def linf_norm_min(params, covariates, target):
    vector = np.abs(target - params.dot(covariates).transpose()) 
    return np.max(vector)

def l1_norm_min(params, covariates, target):
    vector = np.abs(target - params.dot(covariates).transpose()) 
    return np.sum(vector)

start = datetime.datetime(2021, 1, 1, 0, 0, 0)
end = datetime.datetime(2021, 12, 30, 23, 0, 0)

data = data.loc[start:end, :].copy()
window_size_days = 1

sim_start = start
sim_current = sim_start + datetime.timedelta(days=window_size_days)
sim_end_train = sim_current - datetime.timedelta(hours=1)
results = data.loc[sim_current:, ["demand"]].copy()
ensemble_params = pd.DataFrame(columns=["intercept", "spline", "quadratic"])
for i in range((360 - window_size_days)*24):
    # calculate loss
    data_window = data.loc[sim_start:sim_current, :].copy()
    # regression approach
    model = sm.OLS(data_window.loc[:sim_end_train, "demand"], pf.fit_transform(data_window.loc[:sim_end_train, ["spline", "quadratic"]])).fit()
    forecast_regression = model.predict(pf.fit_transform(np.array(data_window.loc[sim_current:, ["spline", "quadratic"]]).reshape(1, -1))).tolist()
    results.loc[sim_current, "forecast_reg"] = forecast_regression[0]
    print(model.params.tolist())
    ensemble_params.loc[i, :] = model.params.tolist()
    print(model.summary())

    """spline_mean_residual = np.mean(np.abs(data_window.loc[:, "demand"] - data_window.loc[:, "spline"]))
    quadratic_mean_residual = np.mean(np.abs(data_window.loc[:, "demand"] - data_window.loc[:, "quadratic"]))
    spline_weight = 1 - spline_mean_residual / (spline_mean_residual + quadratic_mean_residual)
    quadratic_weight = 1 - spline_weight
    results.loc[sim_current, "forecast_combined_means"] = spline_weight * data_window.loc[sim_current, "spline"] + quadratic_weight * data_window.loc[sim_current, "quadratic"]"""
    # l_inf norm approach
    """covariates = np.array(data_window.loc[:, ["spline", "quadratic"]]).transpose()
    target = np.array(data_window.loc[:, "demand"])
    params = model.params.tolist()[1:]
    res = opt.minimize(l1_norm_min, x0=params, args=(covariates, target), method='BFGS')
    print(res.x)
    results.loc[sim_current, "forecast_linf"] = res.x.dot(np.array(data_window.loc[sim_current, ["spline", "quadratic"]]).transpose())"""
    #results.loc[sim_current, "forecast_reg"] = 0.5 * data_window.loc[sim_current, "spline"] + 0.5 * data_window.loc[sim_current, "quadratic"]

    sim_start = sim_start + datetime.timedelta(hours=1)
    sim_current = sim_current + datetime.timedelta(hours=1)
    sim_end_train = sim_end_train + datetime.timedelta(hours=1)

print(ensemble_params)
print(results)
plt.plot(results.loc[:, "demand"], label="Demand")
plt.plot(results.loc[:, "forecast_reg"], label="Regression model")
plt.legend()
plt.show()

"""plt.plot(ensemble_params.loc[:, ["spline", "quadratic", "queb"]], label=["Regression splines", "Fourier series", "Quebec temperature"])
plt.legend()
plt.show()"""

print("RMSE for combined forecast: ")
print(rmse(np.array(results.loc[:, "demand"]), np.array(results.loc[:, "forecast_reg"])))

print("MAPE for combined forecast: ")
print(mape(np.array(results.loc[:, "demand"]), np.array(results.loc[:, "forecast_reg"])))

residuals = abs(results.loc[:, "demand"] - results.loc[:, "forecast_reg"])
plt.hist(residuals, bins=100)
plt.show()

# 500 MW success rate
print("500 MW success rate: ")
print(len(residuals[residuals <= 500]) / len(residuals))

print("1000 MW success rate: ")
print(len(residuals[residuals <= 1000]) / len(residuals))

results.to_csv("results\combined_spain.csv")



"""target = np.array(data.loc[:, "demand"])
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

"""