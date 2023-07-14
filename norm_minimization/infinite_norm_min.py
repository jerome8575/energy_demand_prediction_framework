import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import scipy.optimize as opt
from statsmodels.tools.eval_measures import rmse
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler

import sys
sys.path.insert(0, "C:\\Users\\jerom\\coding\\energy_demand_prediction_framework\\")
sys.path.insert(0, "C:\\Users\\jerom\\energy_demand_prediction_framework")


sr_results = pd.read_csv("results\simulation_results_spline_regression.csv")
quad_results = pd.read_csv("results\simulation_results_quadratic_regression.csv")

data = pd.DataFrame(columns=["spline", "quadratic", "actual"])
data["spline"] = sr_results.loc[:, "forecast"]
data["quadratic"] = quad_results.loc[:, "forecast"]
data["actual"] = sr_results.loc[:, "demand"]
data["date_time"] = sr_results.loc[:, "date_time"]
data.set_index("date_time", inplace=True)


#lin reg 
pf = PolynomialFeatures(degree=1)
model = sm.OLS(data.loc[:, "actual"], pf.fit_transform(data.loc[:, ["spline", "quadratic"]])).fit()
forecast_reg = model.predict(pf.fit_transform(data.loc[:, ["spline", "quadratic"]]))

def l2_norm_min(params):
    return np.sum((data.loc[:, "actual"] - params[0] * data.loc[:, "spline"] - params[1] * data.loc[:, "quadratic"] - params[2])**8)

def l1_norm_min(params):
    return np.sum(np.abs(data.loc[:, "actual"] - params[0] * data.loc[:, "spline"] - params[1] * data.loc[:, "quadratic"] - params[2]))

result = opt.minimize(l2_norm_min, x0=[0.5, 0.5, 0.5], method="Nelder-Mead")

print(result.fun)
print(result.x)

forecast = result.x[0] * data.loc[:, "spline"] + result.x[1] * data.loc[:, "quadratic"] + result.x[2]
actual = data.loc[:, "actual"]

print(rmse(forecast, actual))
print(rmse(forecast_reg, actual))

residuals = actual - forecast

print("Percentage within 1000 mwh")
print(sum(list(map(lambda x: int(x <= 1000), abs(residuals))))/len(residuals))

print("Percentage within 500 mwh")
print(sum(list(map(lambda x: int(x <= 500), abs(residuals))))/len(residuals))


"""
plt.plot(forecast, label="forecast")
plt.plot(actual, label="actual")
plt.legend()
plt.show()"""
