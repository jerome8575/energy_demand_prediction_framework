import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tools.eval_measures import rmse
from sklearn.metrics import mean_absolute_percentage_error as mape
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler

start = datetime.datetime(2021, 1, 1, 0, 0, 0)
end = datetime.datetime(2021, 12, 31, 23, 0, 0)

spline_data = pd.read_csv("results\simulation_results_spline.csv")
spline_data["date_time"] = list(map(lambda t: datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S'), spline_data.loc[:, "date_time"]))
spline_data.set_index("date_time", inplace=True)

spline_data = spline_data.loc[start:end, :].copy()

quad_data = pd.read_csv("results\simulation_results_quad.csv")
quad_data["date_time"] = list(map(lambda t: datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S'), quad_data.loc[:, "date_time"]))
quad_data.set_index("date_time", inplace=True)

quad_data = quad_data.loc[start:end, :].copy()

# plot both forecasts

"""plt.plot(spline_data.loc[:, "demand"], label="Demand")
plt.plot(spline_data.loc[:, "forecast"], label="Spline Forecast")
plt.plot(quad_data.loc[:, "forecast"], label="Quadratic Forecast")
plt.legend()
plt.show()"""

# combine forecasts by hour

spline_data["hour"] = spline_data.index.hour
quad_data["hour"] = quad_data.index.hour

forecasts = pd.DataFrame({})

for h in range(24):

    spline_data_hour = spline_data[spline_data.loc[:, "hour"] == h].copy()
    quad_data_hour = quad_data[quad_data.loc[:, "hour"] == h].copy()

    fits = pd.DataFrame({
        "spline": spline_data_hour.loc[:, "forecast"],
        "quad": quad_data_hour.loc[:, "forecast"],
    })

    model = sm.OLS(endog=np.array(spline_data_hour.loc[:, "demand"]), exog=fits).fit()

    forecasts_hour = model.predict(fits).tolist()
    forecasts["hour_" + str(h)] = forecasts_hour

print(forecasts)
forecasts = np.array(forecasts).flatten()

print("RMSE")
print(rmse(np.array(spline_data.loc[:, "demand"]), forecasts))

print("MAPE")
print(mape(np.array(spline_data.loc[:, "demand"]), forecasts))

plt.plot(np.array(spline_data.loc[:, "demand"]), label="Demand")
plt.plot(forecasts, label="Combined Forecast")
plt.legend()
plt.show()

