import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from prophet import Prophet
import yaml
from pydantic import BaseModel
from typing import Union, List


class Site(BaseModel):
    name: str
    file_training: Path
    file_combined: Path


class Template(BaseModel):
    name: str
    file: Path


class Config(BaseModel):
    data_folder: Path
    sites: List[Site]
    templates: List[Template]


def load_yaml(path: Path) -> Union[dict, None]:
    """Load a yaml file form path and return dict"""
    with open(path, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return None


config = load_yaml("config.yaml")
config = Config(**config)

rng = np.random.RandomState(42)

template = pd.read_csv(
    config.data_folder
    / next(template.file for template in config.templates if template.name == "phase1"),
    # parse_dates=["date"],
    # dayfirst=True,
)


for site in config.sites:

    X = pd.read_csv(
        config.data_folder / site.file_training,
        parse_dates=["time", "maxtime", "mintime"],
        dayfirst=True,
    )

    X.rename(columns={"time": "ds", "maxvalue": "y"}, inplace=True)

    m = Prophet(yearly_seasonality=True)
    m.fit(X)

    df = pd.read_csv(
        config.data_folder / site.file_combined,
        parse_dates=["time"],
        dayfirst=True,
    )
    forecast = m.predict(df.rename(columns={"time": "ds"}))
    forecast = forecast.merge(df, left_on="ds", right_on="time")
    forecast["charger"] = forecast.value - forecast.yhat
    forecast.set_index("time", inplace=True)
    forecast["charger_daily_max"] = forecast.resample("D").charger.transform(max)

    # plot something
    size = 1000
    figsize = (12, 6)
    forecast.charger.head(size).plot(figsize=figsize, label="Charger")
    forecast.charger_daily_max.head(size).plot(
        figsize=figsize, label="Charger Daily Max"
    )
    forecast.yhat.head(size).plot(figsize=figsize, label="Substation")
    forecast.value.head(size).plot(figsize=figsize, label="Combined")
    plt.legend()
    plt.show()

    # Generate output for scoring
    daily = forecast.resample("D").charger.max().to_frame()
    template_dates = pd.to_datetime(template.date.values, dayfirst=True)
    mask = (daily.index >= template_dates.min()) & (daily.index <= template_dates.max())
    daily = daily.loc[mask]
    # daily["site"] = site.name
    # daily.reset_index(inplace=True)

    mask = template.substation == site.name
    template.loc[mask, "value"] = daily.charger.values


template.to_csv("data/submission.csv", index=False)
