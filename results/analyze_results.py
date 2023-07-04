import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt

def calc_mape(df):
    return np.mean(np.abs((np.array(df.loc[:, "demand"]) - np.array(df.loc[:, "forecast"]))) / np.array(df.loc[:, "demand"])) * 100


start = datetime.datetime(2021, 1, 1, 0, 0, 0)
end = datetime.datetime(2021, 12, 31, 23, 0, 0)

data = pd.read_csv("results\combined_forecast.csv")
data["date_time"] = list(map(lambda t: datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S'), data.loc[:, "date_time"]))
data.set_index("date_time", inplace=True)

data = data.loc[start:end, :].copy()

demand = np.array(data.loc[:, "demand"])
forecasts = np.array(data.loc[:, "forecast"])
"""
plt.scatter(data.loc[:, "scaled_temp"], data.loc[:, "demand"])
plt.scatter(data.loc[:, "scaled_temp"], data.loc[:, "forecast"])
plt.show()"""


mape = calc_mape(data)
print("MAPE")
print(mape)

mapes = []
for i in range(1, 12):
    start = datetime.datetime(2021, i, 1, 0, 0, 0)
    end = datetime.datetime(2021, i+1, 1, 0, 0, 0) - datetime.timedelta(hours=1)
    mapes.append(calc_mape(data.loc[start:end, :]))

plt.plot(mapes)
plt.show()

residuals = demand - forecasts

print("Percentage within 1000 mwh")
print(sum(list(map(lambda x: int(x <= 1000), abs(residuals))))/len(residuals))

print("Percentage within 500 mwh")
print(sum(list(map(lambda x: int(x <= 500), abs(residuals))))/len(residuals))

plt.plot(data.loc[:, ["demand", "forecast"]])
plt.show()

plt.hist(residuals, bins=100)
plt.show()

plt.plot(residuals)
plt.show()

