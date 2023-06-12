import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt

start = datetime.datetime(2022, 6, 15, 0, 0, 0)
end = datetime.datetime(2022, 8, 30, 23, 0, 0)

data = pd.read_csv("results\simulation_results.csv")
data["date_time"] = list(map(lambda t: datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S'), data.loc[:, "date_time"]))
data.set_index("date_time", inplace=True)

demand = np.array(data.loc[start:end, "demand"])
forecasts = np.array(data.loc[start:end, "forecast"])

mape = np.mean(np.abs((demand - forecasts) / demand)) * 100
print("MAPE")
print(mape)

residuals = demand - forecasts

print("Percentage within 1000 mwh")
print(sum(list(map(lambda x: int(x <= 1000), abs(residuals))))/len(residuals))

print("Percentage within 500 mwh")
print(sum(list(map(lambda x: int(x <= 500), abs(residuals))))/len(residuals))

plt.plot(data.loc[start:end, ["demand", "forecast"]])
plt.show()

plt.hist(residuals, bins=100)
plt.show()

