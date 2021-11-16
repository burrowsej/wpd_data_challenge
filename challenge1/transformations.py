from datetime import datetime, timedelta
import calendar

import plotly.express as px
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance
from scipy.signal import hilbert


DAYS_PER_WEEK = 7

# these are the final feature and target column names
features = [
    f"{op}({col})"
    for col in ("fraction_of_day", "fraction_of_week", "fraction_of_year")  # , "wind_orientation")
    for op in ("sin", "cos")
]
features.extend(
    (
        "value",
        "first_derivative",
        "second_derivative",
        "third_derivative",
        "fourth_derivative",
        "local_bandwidth",
        "year",
        "temperature",
        "solar_irradiance",
        # "windspeed",
        "windspeed_north",
        "windspeed_east",
        "pressure",
        "spec_humidity",
    )
)
targets = ["value_max", "value_min"]


# weather data is lower resolution than power data
def upsample(df):
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").resample("30Min").interpolate(method="linear")
    df = df.reset_index()
    df = df.rename(columns={"datetime": "time"})
    df["time"] = df["time"].astype(str)
    return df


def join_x(df_x1, df_x2):
    df = pd.merge(
        left=df_x1,
        right=df_x2,
        on="time",
        how="left",  # this might cause issues
    )
    return df


def join_y(df_x, df_y):
    df = pd.merge(
        left=df_x,
        right=df_y,
        on="time",
        how="left",  # this might cause issues
    )
    return df


# main feature is time - likely 3 scales of variation: daily, weekly, annually (seasonal)
def prepare(df):
    df = df.copy()
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
    # df["windspeed"] = np.sqrt(df["windspeed_north"] ** 2 + df["windspeed_east"] ** 2)
    # df["wind_orientation"] = np.arctan2(df["windspeed_north"], df["windspeed_east"])
    # df["wind_orientation"] = (
    #    df["wind_orientation"] % (2 * np.pi) / (2 * np.pi)
    # )  # put it in range [0, 1) like others
    df = df.drop(columns=["time", "month", "day"])  # , "windspeed_north", "windspeed_east"])
    return df


# temporal features are all cyclic - encode them as such
# nb: no concern about leakage before splitting into train/valid bc encoder doesn't need to be fit
def encode(df):
    df = df.copy()
    for col in (
        "fraction_of_day",
        "fraction_of_week",
        "fraction_of_year",
    ):  # , "wind_orientation"):
        df[f"sin({col})"] = np.sin(2 * np.pi * df[col])
        df[f"cos({col})"] = np.cos(2 * np.pi * df[col])
        df = df.drop(columns=col)
    return df


# nb: assumes data is ordered and equally spaced
# nb: gradient is negated? curvature may be too? something to fix, but unlikely to affect outputs
# nb: all second-order accurate kernels (unsure about lop-sided ones...)
# coefficients from https://en.wikipedia.org/wiki/Finite_difference_coefficient
# and from https://web.media.mit.edu/~crtaylor/calculator.html
def calculate_geometric_features(df):
    df = df.copy()
    kernel_f = np.array([-3 / 2, 2, -1 / 2])  # forward difference
    kernel_c = np.array([-1 / 2, 0, 1 / 2])  # central difference
    kernel_b = np.array([1 / 2, -2, 3 / 2])  # backward difference
    df["first_derivative"] = np.concatenate(
        (
            [np.dot(df.iloc[:3]["value"], kernel_f)],
            np.convolve(df["value"], kernel_c, "valid"),
            [np.dot(df.iloc[-3:]["value"], kernel_b)],
        )
    )
    kernel_f = np.array([2, -5, 4, -1])  # forward difference
    kernel_c = np.array([1, -2, 1])  # central difference
    kernel_b = np.array([-1, 4, -5, 2])  # backward difference
    df["second_derivative"] = np.concatenate(
        (
            [np.dot(df.iloc[:4]["value"], kernel_f)],
            np.convolve(df["value"], kernel_c, "valid"),
            [np.dot(df.iloc[-4:]["value"], kernel_b)],
        )
    )
    kernel_f = np.array([-5 / 2, 9, -12, 7, -3 / 2])  # forward difference
    kernel_fc = np.array([-3 / 2, 5, -6, 3, -1 / 2])  # lop-sided difference
    kernel_c = np.array([-1 / 2, 1, 0, -1, 1 / 2])  # central difference
    kernel_bc = np.array([1 / 2, -3, 6, -5, 3 / 2])  # lop-sided difference
    kernel_b = np.array([3 / 2, -7, 12, -9, 5 / 2])  # backward difference
    df["third_derivative"] = np.concatenate(
        (
            [np.dot(df.iloc[:5]["value"], kernel_f)],
            [np.dot(df.iloc[1:6]["value"], kernel_fc)],
            np.convolve(df["value"], kernel_c, "valid"),
            [np.dot(df.iloc[-6:-1]["value"], kernel_bc)],
            [np.dot(df.iloc[-5:]["value"], kernel_b)],
        )
    )
    kernel_f = np.array([2, 3, -14, 26, -24, 11, -2])  # forward difference
    kernel_fc = np.array([5, -36, 111, -184, 171, -84, 17]) / 6  # lop-sided difference
    kernel_c = np.array([1, -4, 6, -4, 1])  # central difference
    kernel_bc = np.array([17, -84, 171, -184, 111, -36, 5]) / 6  # lop-sided difference
    kernel_b = np.array([-2, 11, -24, 26, -14, 3, 2])  # backward difference
    df["fourth_derivative"] = np.concatenate(
        (
            [np.dot(df.iloc[:7]["value"], kernel_f)],
            [np.dot(df.iloc[1:8]["value"], kernel_fc)],
            np.convolve(df["value"], kernel_c, "valid"),
            [np.dot(df.iloc[-8:-1]["value"], kernel_bc)],
            [np.dot(df.iloc[-7:]["value"], kernel_b)],
        )
    )
    win = np.ones(len(df))
    win[:4] = np.blackman(8)[:4]
    win[-4:] = np.blackman(8)[-4:]
    v = hilbert(df["value"] * win)
    kernel_f = np.array([-3 / 2, 2, -1 / 2])  # forward difference
    kernel_c = np.array([-1 / 2, 0, 1 / 2])  # central difference
    kernel_b = np.array([1 / 2, -2, 3 / 2])  # backward difference
    dvdt = np.concatenate(
        (
            [np.dot(v[:3], kernel_f)],
            np.convolve(v, kernel_c, "valid"),
            [np.dot(v[-3:], kernel_b)],
        )
    )
    df["local_bandwidth"] = np.abs(dvdt)
    return df


# split out final 12 months for validation
# nb: these are pre-sorted by date within the `prepare`` function
def split(df):
    df = df.copy()
    final_year = df["year"].max()
    days_in_final_year = 366 if calendar.isleap(final_year) else 365
    mask_train = df["datetime"] < (df["datetime"].max() - timedelta(days=days_in_final_year))
    df = df.drop(columns="datetime")
    df_train = df[mask_train]
    df_valid = df[~mask_train]
    return (df_train, df_valid)


def train(df):
    X = df[features]
    ys = df[targets].values
    forward = lambda ys: np.vstack((0.5 * ys.sum(axis=1), ys[:, 1] - ys[:, 0])).T
    backward = lambda ys: np.vstack((ys[:, 0] - 0.5 * ys[:, 1], ys[:, 0] + 0.5 * ys[:, 1])).T
    regr = TransformedTargetRegressor(
        MultiOutputRegressor(RandomForestRegressor()),
        func=forward,
        inverse_func=backward,
    )
    regr.fit(X, ys)
    return regr


# score is sum of root mean squared errors for each target
def validate(regr, df):
    df = df.copy()
    X = df[features]
    df[[f"pred({col})" for col in targets]] = regr.predict(X)
    df = df.rename(columns={col: f"true({col})" for col in targets})
    df = df.drop(columns=features)
    return df


def score(df):
    score = 0
    for col in targets:
        score += mean_squared_error(df[f"true({col})"], df[f"pred({col})"], squared=True)
    return np.sqrt(score)


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


# benchmark assumes min=max=value
def benchmark(df):
    df = df.copy()
    for col in targets:
        df[f"pred({col})"] = df["value"]
        df = df.rename(columns={col: f"true({col})"})
    df = df.drop(columns=features)
    return df


# skill is ratio of model RMSE and benchmark RMSE
def skill(score_model, score_benchmark):
    score = score_model / score_benchmark
    print(score)
    return score


def submit(template, df_pred):
    template = template.iloc[:-1]
    df_pred = df_pred.copy()
    df_pred = df_pred.rename(
        columns={
            "datetime": "time",
            "pred(value_max)": "value_max",
            "pred(value_min)": "value_min",
        }
    )
    df_pred["time"] = df_pred["time"].apply(str)
    df = pd.merge(
        left=template.drop(columns=["value_min", "value_max"]),
        right=df_pred,
        on="time",
        how="left",
        validate="1:1",
    )
    return df


def feature_importance(regr, df):
    X = df[features]
    ys = df[targets]
    fi = permutation_importance(regr, X, ys)
    px.bar(
        x=features,
        y=fi["importances_mean"],
        log_y=True,
    ).show()
    print(fi)
    afsdfdsfsd
