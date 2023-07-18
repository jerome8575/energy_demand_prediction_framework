import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tools.eval_measures import rmse
from sklearn.metrics import mean_absolute_percentage_error as mape

def calc_mape(df):
    return np.mean(np.abs((np.array(df.loc[:, "demand"]) - np.array(df.loc[:, "forecast"]))) / np.array(df.loc[:, "demand"])) * 100


start = datetime.datetime(2021, 1, 1, 0, 0, 0)
end = datetime.datetime(2021, 12, 31, 23, 0, 0)

data = pd.read_csv("results\quadratic_fourier_regression_2021.csv")
data["date_time"] = list(map(lambda t: datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S'), data.loc[:, "date_time"]))
data.set_index("date_time", inplace=True)

data = data.loc[start:end, :].copy()

hour = 15 
data_hour = data[data.index.hour == hour].copy()

plt.plot(data_hour.loc[:, "demand"], label="Demand")
plt.plot(data_hour.loc[:, "forecast"], label="Forecast")
plt.legend()
plt.show()

"""plt.scatter(data.loc[:, "scaled_temp"], data.loc[:, "demand"])
plt.scatter(data.loc[:, "scaled_temp"], data.loc[:, "forecast"])
plt.show()"""


mape = calc_mape(data)
print("MAPE")
print(mape)



start_march = datetime.datetime(2021, 3, 1, 0, 0, 0)
end_march = datetime.datetime(2021, 3, 31, 23, 0, 0)
print("MAPE for March")
print(calc_mape(data.loc[start_march:end_march, :]))

mapes = []
for i in range(1, 12):
    start = datetime.datetime(2021, i, 1, 0, 0, 0)
    end = datetime.datetime(2021, i+1, 1, 0, 0, 0) - datetime.timedelta(hours=1)
    mapes.append(calc_mape(data.loc[start:end, :]))

plt.plot(mapes)
plt.title("MAPE for each month")
plt.xlabel("Month")
plt.ylabel("MAPE")
plt.show()

residuals = data.loc[:, "demand"] - data.loc[:, "forecast"]

print("Percentage within 1000 mwh")
print(sum(list(map(lambda x: int(x <= 1000), abs(residuals))))/len(residuals))

print("Percentage within 500 mwh")
print(sum(list(map(lambda x: int(x <= 500), abs(residuals))))/len(residuals))

plt.plot(data.loc[:, "demand"], label="Demand")
plt.plot(data.loc[:, "forecast"], label="Forecast")
plt.legend()
plt.show()

print(rmse(data.loc[:, "demand"], data.loc[:, "forecast"]))

"""plt.plot(data.loc[:, "demand"] - data.loc[:, "forecast"], label="Residuals")
plt.legend()
plt.show()
"""
plt.hist(residuals, bins=100)
plt.show()

plt.plot(residuals)
plt.show()

