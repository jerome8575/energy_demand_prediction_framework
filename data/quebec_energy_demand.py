
import pandas as pd
import numpy as np
import datetime 

class HQ_data():

    def __init__(self):

        self.start = datetime.datetime(2019, 1, 1, 1, 0, 0)
        self.end = datetime.datetime(2022, 12, 31, 23, 0, 0)
        data_2019 = pd.read_excel("data_files\\2019-demande-electricite-quebec.xlsx")
        data_2020 = pd.read_excel("data_files\\2020-demande-electricite-quebec.xlsx")
        data_2021 = pd.read_excel("data_files\\2021-demande-electricite-quebec.xlsx")
        data_2022 = pd.read_excel("data_files\\2022-demande-electricite-quebec.xlsx")

        demand_data = pd.concat([data_2019, data_2020, data_2021, data_2022])
        weather_data = pd.read_csv("data_files\quebec_weather_index.csv")

        self.data = self.get_features(demand_data, weather_data)

    def get_features(self, demand_data, weather_data):
        demand_data.set_index("Date", inplace=True)
        datetime_index = list(map(lambda t: datetime.datetime.strptime(t, '%Y-%m-%d %H:%M'), weather_data.loc[:, "datetime"]))

        # get weather data
        weather_data["datetime_index"] = datetime_index
        weather_data.set_index("datetime_index", inplace=True)

        # get type of day (weekday or weekend)
        print(weather_data.loc[self.start:self.end, "temp"])
        print(demand_data.loc[self.start:self.end, "Moyenne (MW)"])
        data = pd.DataFrame({
            "demand": list(demand_data.loc[self.start:self.end, "Moyenne (MW)"]),
            #"temp": list(weather_data.loc[self.start:self.end, "temp"])
            "date_time": list(demand_data.index[0:-1])
        })
        data.set_index("date_time", inplace=True)
        data.fillna(method="backfill", inplace=True)
        return data
    
    def get_history(self, start, end):

        return self.data.loc[start:end, :]
    