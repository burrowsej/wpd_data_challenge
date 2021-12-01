from pathlib import Path
import numpy as np
import pandas as pd
import calendar
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.inspection import permutation_importance
from sklearn.multioutput import MultiOutputRegressor
from sklearn.compose import TransformedTargetRegressor
from functools import lru_cache
from findiff import FinDiff


def combine_features_targets(features: Path, targets: Path) -> pd.DataFrame:
    basetable = pd.read_csv(
        features,
        parse_dates=["time"],
        index_col="time",
        dayfirst=True,
    )

    targets = pd.read_csv(
        targets,
        parse_dates=["time"],
        index_col="time",
        dayfirst=True,
    )

    basetable = basetable.merge(targets, left_index=True, right_index=True)
    return basetable


def engineer_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts temporal features from a datetime index"""

    df["year"] = df.index.year

    df["annual_quantile"] = df.index.map(
        lambda x: x.dayofyear / (366 if calendar.isleap(x.year) else 365)
    )

    df["weekly_quantile"] = df.index.dayofweek

    df["hour"] = df.index.hour + df.index.minute / 60

    # encode cyclical features in two separate sin and cos transforms
    max_vals = {"annual_quantile": 366, "weekly_quantile": 6, "hour": 23.5}
    for col in max_vals:
        df[f"sin_{col}"] = np.sin((2 * np.pi * df[col]) / max_vals[col])
        df[f"cos_{col}"] = np.cos((2 * np.pi * df[col]) / max_vals[col])
        # df.plot.scatter(f"sin_{col}", f"cos_{col}").set_aspect("equal")
        df.drop(columns=col, inplace=True)

    return df


def get_rmse(day: pd.DataFrame) -> float:
    """Fit a polynomial and get the rmse to quantify noise for the day"""
    X = np.c_[day.index.hour + day.index.minute / 60]
    y = day.value.values

    polyreg = make_pipeline(PolynomialFeatures(degree=7), LinearRegression())
    polyreg.fit(X, y)

    rmse = np.sqrt(mean_squared_error(y, polyreg.predict(X)))

    return rmse


def engineer_30min_demand_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts features form 30min demand data"""

    # forward and backward diffs
    # TODO: replace this with Elliot's finite difference thing
    # df["diff_forwards"] = df.value.diff(periods=-1).ffill()
    # df["diff_backwards"] = df.value.diff(periods=1).bfill()

    # finite difference method to calculate derivatives
    accuracy = 2
    for d in range(1, 6):
        derivative = FinDiff(0, accuracy, d)
        df[f"d{d}_value"] = derivative(df.value)
    #     df[f"d{d}_value"].head(50).plot()
    # plt.legend()
    # plt.rcParams["figure.figsize"] = (12, 6)
    # plt.show()

    # daily noise
    df["date"] = df.index.date
    rmse_by_day = df.groupby("date").apply(get_rmse)
    rmse_by_day = pd.Series(rmse_by_day, name="daily_noise")
    df.drop(columns="date", inplace=True)

    df = df.merge(rmse_by_day, left_index=True, right_index=True, how="left")
    df.daily_noise.ffill(inplace=True)

    return df


def engineer_weather_features(df: pd.DataFrame, weather_path: Path) -> pd.DataFrame:
    """Extracts features from weather data and joins it to basetable"""
    weather = pd.read_csv(
        weather_path,
        parse_dates=["datetime"],
        index_col="datetime",
        dayfirst=True,
    )

    # resample to 30mins to get same as demand data
    weather = weather.resample("30T").mean()
    weather = weather.interpolate()

    # derivatives of irradiance
    # accuracy = 2
    # for d in range(1, 6):
    #     derivative = FinDiff(0, accuracy, d)
    #     weather[f"d{d}_solar_irradiance"] = derivative(weather.solar_irradiance)

    # vectorise wind speed components
    weather["windspeed"] = weather.apply(
        lambda x: np.sqrt(x.windspeed_north ** 2 + x.windspeed_east ** 2), axis=1
    )
    weather["winddirection"] = weather.apply(
        lambda x: np.arctan2(x.windspeed_north, x.windspeed_east), axis=1
    )

    # TODO: read up on this and explore - should not need the pis and the 2s
    weather["sin_winddirection"] = np.sin(
        (2 * np.pi * weather["winddirection"]) / np.pi
    )
    weather["cos_winddirection"] = np.cos(
        (2 * np.pi * weather["winddirection"]) / np.pi
    )
    # weather.plot.scatter("sin_winddirection", "cos_winddirection").set_aspect("equal")
    weather.drop(columns="winddirection", inplace=True)

    df = df.merge(weather, left_index=True, right_index=True, how="left")

    return df


def train(df, targets):
    """Transforms the targets from max, min to var and skew, then fits
    random forest regressor"""
    features = [col for col in df.columns if col not in targets]

    X = df[features]
    ys = df[targets].values
    forward = lambda ys: np.vstack(
        (
            # skew component (average which can be compared to value)
            0.5 * ys.sum(axis=1),
            # dispersion component (range)
            ys[:, 1] - ys[:, 0],
        )
    ).T
    backward = lambda ys: np.vstack(
        (ys[:, 0] - 0.5 * ys[:, 1], ys[:, 0] + 0.5 * ys[:, 1])
    ).T
    reg = TransformedTargetRegressor(
        MultiOutputRegressor(RandomForestRegressor()),
        func=forward,
        inverse_func=backward,
    )
    reg.fit(X, ys)
    return reg


def get_Xy(basetable: pd.DataFrame, target: str = "max"):
    X = basetable.drop(columns=["value_max", "value_min"]).values
    if target == "max":
        y = basetable.value_max.values
    elif target == "min":
        y = basetable.value_min.values
    else:
        raise ValueError
    return X, y


def get_feature_importance(reg, X, y, df) -> pd.DataFrame():

    r = permutation_importance(reg, X, y, n_repeats=30, random_state=7)

    cols = df.columns.to_list()
    feature_names = cols[:1] + cols[3:]
    features_importance = pd.DataFrame(
        {
            "mean_importance": r.importances_mean,
            "std_importance": r.importances_std,
        },
        index=feature_names,
    )

    features_importance.sort_values(by="mean_importance", inplace=True)

    features_importance.mean_importance.plot.barh(
        logx=True,
        figsize=(10, 5),
        title="Feature importance",
    )

    return features_importance
