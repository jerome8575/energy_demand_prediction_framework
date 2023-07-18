import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class RE_data:

    def __init__(self) -> None:

        scaler = StandardScaler()

        start = datetime.datetime(2016, 1, 1, 0, 0, 0)
        end = datetime.datetime(2022, 12, 31, 23, 0, 0)

        # read spain demand data
        self.data = pd.read_csv("data_files\\spain_demand_hourly_aggregate.csv")
        self.data.columns = ["date_time", "demand"]
        self.data["date_time"] = list(map(lambda t: datetime.datetime.strptime(str(t), '%Y-%m-%d %H:%M:%S'), self.data.loc[:, "date_time"]))
        self.data.set_index("date_time", inplace=True)
        self.data = self.data.loc[start:end, :]
        self.data.fillna(method="ffill", inplace=True)

        # read spain weather data
        spain_temp = pd.read_csv("data_files\\Madrid_40_416775_-3_70379_646778322819b9000923cb9a.csv")
        date_time = list(map(lambda t: datetime.datetime.strptime(str(t[0:19]), '%Y-%m-%d %H:%M:%S'), spain_temp.loc[:, "dt_iso"]))
        spain_temp["date_time"] = date_time
        spain_temp.set_index("date_time", inplace=True)

        spain_temp = spain_temp.loc[start:end, ["temp", "clouds_all", "humidity"]]
        spain_temp["datetime"] = spain_temp.index
        spain_temp.drop_duplicates(inplace=True)

        temp = list(spain_temp.loc[:, "temp"])
        clouds = list(spain_temp.loc[:, "clouds_all"])
        humidity = list(spain_temp.loc[:, "humidity"])

        self.data["temp"] = temp
        self.data["scaled_temp"] = scaler.fit_transform(self.data.loc[:, ["temp"]])

        self.data["log_demand"] = np.log(self.data.loc[:, "demand"])
        self.data["day"] = list(map(lambda t: t.weekday(), self.data.index))
        self.data["hour"] = self.data.index.hour
        self.data["is_clear"] = clouds
        self.data["rel_hum"] = humidity

        for i in range(1, 24):
            self.data["temp_" + str(i)] = [0]*i + list( self.data["scaled_temp"])[:-i]
            self.data["temp_index_" + str(i)] = (np.array(self.data.loc[:, "temp_" + str(i)]) * np.array(self.data.loc[:, "scaled_temp"]))
        
        self.data["is_weekend"] = list(map(lambda t: 1 if t.weekday() > 4 else 0, self.data.index))

        self.data["temp_24"] = [0]*24 + list( self.data["scaled_temp"])[:-24]
        self.data["temp_36"] = [0]*36 + list( self.data["scaled_temp"])[:-36]
        self.data["temp_48"] = [0]*48 + list( self.data["scaled_temp"])[:-48]
        self.data["temp_60"] = [0]*60 + list( self.data["scaled_temp"])[:-60]
        self.data["temp_72"] = [0]*72 + list( self.data["scaled_temp"])[:-72]
        self.data["temp_84"] = [0]*84 + list( self.data["scaled_temp"])[:-84]
        self.data["temp_96"] = [0]*96 + list( self.data["scaled_temp"])[:-96]

        l = list(self.data.loc[:, "scaled_temp"])
        diff = [l[i] - l[i-24] for i in range(24, len(l))]
        self.data["scaled_temp_diff_24"] = [0] * 24 + diff

        l = list(self.data.loc[:, "scaled_temp"])
        diff = [l[i] - l[i-48] for i in range(48, len(l))]
        self.data["scaled_temp_diff_48"] = [0] * 48 + diff

        self.data["demand_lag_24"] = [0]*24 + list(self.data["log_demand"])[:-24]

    def get_history(self, start = datetime.datetime(2019, 1, 1, 1, 0, 0), end = datetime.datetime(2022, 12, 31, 23, 0, 0)):
        return self.data.loc[start:end, :]