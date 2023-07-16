import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import scipy.optimize as opt
import scipy.linalg as la
from statsmodels.tools.eval_measures import rmse
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler

import sys
sys.path.insert(0, "C:\\Users\\jerom\\coding\\energy_demand_prediction_framework\\")
sys.path.insert(0, "C:\\Users\\jerom\\energy_demand_prediction_framework")

start = datetime.datetime(2021, 2, 1, 0, 0, 0)
end = datetime.datetime(2021, 2, 28, 23, 0, 0)

sr_results = pd.read_csv("results\simulation_results_spline_regression.csv")
quad_results = pd.read_csv("results\simulation_results_quadratic_regression.csv")

data = pd.DataFrame(columns=["spline", "quadratic", "actual"])
data["spline"] = sr_results.loc[:, "forecast"]
data["quadratic"] = quad_results.loc[:, "forecast"]
data["actual"] = sr_results.loc[:, "demand"]
data["date_time"] = list(map(lambda t: datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S'), sr_results.loc[:, "date_time"]))
data.set_index("date_time", inplace=True)

data = data.loc[start:end, :].copy()

#lin reg 
pf = PolynomialFeatures(degree=1)
model = sm.OLS(data.loc[:, "actual"], pf.fit_transform(data.loc[:, ["spline", "quadratic"]])).fit()
forecast_reg = model.predict(pf.fit_transform(data.loc[:, ["spline", "quadratic"]]))

def infinite_norm(params):
    return np.max(np.abs(data.loc[:, "actual"] - params[0] * data.loc[:, "spline"] - params[1] * data.loc[:, "quadratic"] - params[2]))

def l2_norm_min(params):
    return np.sum((data.loc[:, "actual"] - params[0] * data.loc[:, "spline"] - params[1] * data.loc[:, "quadratic"] - params[2])**2)

def l1_norm_min(params):
    return np.sum(np.abs(data.loc[:, "actual"] - params[0] * data.loc[:, "spline"] - params[1] * data.loc[:, "quadratic"] - params[2]))

result = opt.minimize(infinite_norm, x0=[0.5, 0.5, 0.5], method="Nelder-Mead")

print(result.fun)
print(result.x)

forecast = result.x[0] * data.loc[:, "spline"] + result.x[1] * data.loc[:, "quadratic"] + result.x[2]
results = data.loc[:, ["actual"]]
results["forecast"] = forecast
results["forecast_reg"] = forecast_reg

print(rmse(results.loc[:, "forecast"], results.loc[:, "actual"]))
print(rmse(results.loc[:, "forecast_reg"], results.loc[:, "actual"]))

residuals = results.loc[:, "actual"] - results.loc[:, "forecast"]
print("Max inf norm residual")
print(np.max(np.abs(residuals)))

reg_residuals = results.loc[:, "actual"] - results.loc[:, "forecast_reg"]
print("Max reg residual")
print(np.max(np.abs(reg_residuals)))

print("Percentage within 1000 mwh")
print(sum(list(map(lambda x: int(x <= 1000), abs(residuals))))/len(residuals))

print("Percentage within 500 mwh")
print(sum(list(map(lambda x: int(x <= 500), abs(residuals))))/len(residuals))



plt.plot(results.loc[:, ["actual", "forecast", "forecast_reg"]], label=["Actual", "Infinite norm", "Linear regression"])
plt.legend()
plt.show()
