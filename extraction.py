import numpy as np
import pandas as pd


folder = "/home/eloy/Code/scala_exercises/progfun5/observatory/src/main/resources"
stations_path = f"{folder}/stations.csv"
temperatures_path = f"{folder}/2015.csv"

stations = pd.read_csv(stations_path, names=("stn", "wban", "lat", "lon"))
stations.dropna(subset=("lat", "lon"), inplace=True)
stations = stations.replace(to_replace={"stn": np.nan, "wban": np.nan}, value=0).astype({"stn": int, "wban": int})
stations.set_index(["stn", "wban"], inplace=True)

temperatures = pd.read_csv(temperatures_path, names=("stn", "wban", "month", "day", "temperature"),
                           usecols=(0, 1, 4))
temperatures.drop(temperatures[temperatures.temperature == 9999.9].index, inplace=True)
temperatures = temperatures.replace(to_replace={"stn": np.nan, "wban": np.nan}, value=0).astype({"stn": int, "wban": int})
temperatures["temp"] = (temperatures["temperature"] - 32) / 1.8  # convert to celsius
temperatures.drop("temperature", axis=1, inplace=True)
temperatures.set_index(["stn", "wban"], inplace=True)

avg_temp = temperatures.groupby(["stn", "wban"]).agg("mean")

joined = avg_temp.join(stations, how="inner")

print(joined)
