from datetime import datetime, timedelta
import calendar

import kedro_light as kl
import plotly.express as px
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


DAYS_PER_WEEK = 7

# these are the final feature and target column names
features = [
    f"{op}({col})"
    for col in ("fraction_of_day", "fraction_of_week", "fraction_of_year")
    for op in ("sin", "cos")
]
features.extend(("value", "year"))
targets = ["value_max", "value_min"]


def join(df_x, df_y):
    df = pd.merge(
        left=df_x,
        right=df_y,
        on="time",
    )
    return df


# main feature is time - likely 3 scales of variation: daily, weekly, annually (seasonal)
def prepare(df):
    df["datetime"] = pd.to_datetime(df["time"])
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["day"] = df["datetime"].dt.day
    df["fraction_of_day"] = (
        df["datetime"]
        - df.apply(lambda r: datetime(year=r["year"], month=r["month"], day=r["day"]), axis=1)
    ) / timedelta(days=1)
    df["fraction_of_week"] = df["datetime"].dt.dayofweek / DAYS_PER_WEEK
    df["fraction_of_year"] = (
        df["datetime"]
        - df["year"].apply(lambda y: datetime(year=y, month=1, day=1))
        - timedelta(days=1)
    ) / df["year"].apply(lambda y: timedelta(days=366 if calendar.isleap(y) else 365))
    df = df.drop(columns=["time", "month", "day"])
    return df


# temporal features are all cyclic - encode them as such
# nb: no concern about leakage before splitting into train/valid bc encoder doesn't need to be fit
def encode(df):
    for col in ("fraction_of_day", "fraction_of_week", "fraction_of_year"):
        df[f"sin({col})"] = np.sin(2 * np.pi * df[col])
        df[f"cos({col})"] = np.cos(2 * np.pi * df[col])
        df = df.drop(columns=col)
    return df


# split out final 12 months for validation
# nb: these are pre-sorted by date within the `prepare`` function
def split(df):
    final_year = df["year"].max()
    days_in_final_year = 366 if calendar.isleap(final_year) else 365
    mask_train = df["datetime"] < (df["datetime"].max() - timedelta(days=days_in_final_year))
    df = df.drop(columns="datetime")
    df_train = df[mask_train]
    df_valid = df[~mask_train]
    return (df_train, df_valid)


def train(df):
    X = df[features]
    ys = [df[targets[j]] for j in (0, 1)]
    regr_min = RandomForestRegressor()
    regr_max = RandomForestRegressor()
    regr_min.fit(X, ys[0])
    regr_max.fit(X, ys[1])
    return regr_min, regr_max


# score is sum of root mean squared errors for each target
def validate(regr_min, regr_max, df):
    df = df.copy()
    X = df[features]
    for col, regr in zip(targets, (regr_min, regr_max)):
        df[f"pred({col})"] = regr.predict(X)
        df = df.rename(columns={col: f"true({col})"})
    df = df.drop(columns=features)
    return df


def score(df):
    score = 0
    for col in targets:
        score += mean_squared_error(df[f"true({col})"], df[f"pred({col})"], squared=True)
    print(score)
    return score


def plot(df):
    df = df.copy()
    df["index"] = df.index
    df = df.melt(
        id_vars="index",
        value_vars=None,  # all other columns
        var_name="variable",
        value_name="value",
    )
    df["interval"] = df["index"] % 48
    df["day"] = df["index"] // 48
    fig = px.scatter(
        data_frame=df,
        x="interval",
        y="value",
        color="variable",
        animation_frame="day",
        range_x=(df["interval"].min(), df["interval"].max()),
        range_y=(df["value"].min(), df["value"].max()),
    )
    fig.show()
    return fig


io = kl.io(conf_paths="conf", catalog="catalog.yml")
dag = [
    kl.node(func=join, inputs=["raw_train_x", "raw_train_y"], outputs="raw_train"),
    kl.node(func=prepare, inputs="raw_train", outputs="prep_train"),
    kl.node(func=encode, inputs="prep_train", outputs="enc_train"),
    kl.node(func=split, inputs="enc_train", outputs=["train", "valid"]),
    kl.node(func=train, inputs="train", outputs=["regr_min", "regr_max"]),
    kl.node(func=validate, inputs=["regr_min", "regr_max", "valid"], outputs="outputs"),
    kl.node(func=score, inputs="outputs", outputs="score"),
    kl.node(func=plot, inputs="outputs", outputs="figure"),
]
kl.run(dag, io)
