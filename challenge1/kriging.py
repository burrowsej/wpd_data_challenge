from pykrige.uk import UniversalKriging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

DATA_FOLDER = Path("data")

WEATHER_DATA = {
    "STAPLEGROVE1": {
        "path": "df_staplegrove_1_hourly.csv",
        "lat": 51.0,
        "long": -3.125,
    },
    "STAPLEGROVE2": {
        "path": "df_staplegrove_2_hourly.csv",
        "lat": 51.0,
        "long": -2.5,
    },
    "STAPLEGROVE3": {
        "path": "df_staplegrove_3_hourly.csv",
        "lat": 51.5,
        "long": -3.125,
    },
    "STAPLEGROVE4": {
        "path": "df_staplegrove_4_hourly.csv",
        "lat": 51.5,
        "long": -2.5,
    },
    "STAPLEGROVE5": {
        "path": "df_staplegrove_5_hourly.csv",
        "lat": 51.5,
        "long": -3.75,
    },
}


df = pd.DataFrame()
for site in WEATHER_DATA:
    df_site = pd.read_csv(
        DATA_FOLDER / WEATHER_DATA[site]["path"],
        usecols=["datetime", "solar_irradiance"],
        parse_dates=["datetime"],
        dayfirst=True,
    )
    df_site["lat"] = WEATHER_DATA[site]["lat"]
    df_site["long"] = WEATHER_DATA[site]["long"]
    df = pd.concat([df, df_site])


def rescale(data, a=0, b=1.0):
    min_data, max_data = min(data), max(data)
    print(min_data, max_data)
    scale = lambda x: a + (((x - min_data) * (b - a)) / (max_data - min_data))
    return data.apply(scale, a, b)


rescale(df.lat)

df = df[["lat", "long", "datetime", "solar_irradiance"]]
