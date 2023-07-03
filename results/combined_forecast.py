
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

pf = PolynomialFeatures(degree=1)

spline_regression = pd.read_csv("results\spline_regression.csv")
quadratic_regression = pd.read_csv("results\quadratic_regression.csv")
sarimax = pd.read_csv("results\sarimax_results.csv")

data = pd.concat([spline_regression.loc[:, ["date_time", "demand", "forecast"]], quadratic_regression.loc[:, "forecast"], sarimax.loc[:, "forecast"]], ignore_index=True, axis=1)
data.columns = ["date_time", "demand", "spline", "quadratic", "sarimax"]
data["date_time"] = list(map(lambda t: datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S'), data.loc[:, "date_time"]))
data.set_index("date_time", inplace=True)

print(data)

start = datetime.datetime(2022, 5, 1, 0, 0, 0)
end = datetime.datetime(2022, 8, 30, 23, 0, 0)

data = data.loc[start:end, :].copy()


target = np.array(data.loc[:, "demand"])

exog = data.loc[:, ["spline", "quadratic"]]

model = sm.OLS(target, exog).fit()
print(model.summary())
combined_forecast = model.predict(exog).tolist()

results = data.loc[:, ["demand"]]
results["forecast"] = combined_forecast

print("RMSE")
print(rmse(np.array(results.loc[:, "demand"]), np.array(results.loc[:, "forecast"])))


results.to_csv("results\combined_forecast.csv")




