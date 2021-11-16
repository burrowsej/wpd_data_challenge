from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from functools import lru_cache
from findiff import FinDiff


DATA_FOLDER = Path("data")
TRAINING_FEATURES = "MW_Staplegrove_CB905_MW_observation_variable_half_hourly_real_power_MW_pre_august.csv"
TRAINING_TARGETS = "MW_Staplegrove_CB905_MW_target_variable_half_hourly_max_min_real_power_MW_pre_august.csv"
AUGUST_FEATURES = (
    "MW_Staplegrove_CB905_MW_observation_variable_half_hourly_real_power_MW_august.csv"
)
AUGUST_TEMPLATE = "Submission_template_august.csv"
SEPTEMBER_FEATURES = "MW_Staplegrove_CB905_MW_observation_variable_half_hourly_real_power_MW_september.csv"
SEPTEMBER_TEMPLATE = "Submisson_template_september.csv"
WEATHER = "df_staplegrove_1_hourly.csv"


def combine_features_targets(features: Path, targets: Path) -> pd.DataFrame:
    basetable = pd.read_csv(
        DATA_FOLDER / features,
        parse_dates=["time"],
        index_col="time",
        dayfirst=True,
    )

    targets = pd.read_csv(
        DATA_FOLDER / targets,
        parse_dates=["time"],
        index_col="time",
        dayfirst=True,
    )

    basetable = basetable.merge(targets, left_index=True, right_index=True)
    return basetable


def engineer_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts temporal features from a datetime index"""

    df["year"] = df.index.year + df.index.month / 12
    df["month"] = df.index.month
    df["dayofweek"] = df.index.dayofweek
    df["hour"] = df.index.hour + df.index.minute / 60

    # encode cyclical features in two separate sin and cos transforms
    max_vals = {"month": 12, "dayofweek": 6, "hour": 23.5}
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

    weather["windspeed"] = weather.apply(
        lambda x: np.sqrt(x.windspeed_north ** 2 + x.windspeed_east ** 2), axis=1
    )
    weather["winddirection"] = weather.apply(
        lambda x: np.arctan2(x.windspeed_north, x.windspeed_east), axis=1
    )

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


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer the full feature set"""
    df = engineer_temporal_features(df)
    df = engineer_30min_demand_features(df)
    df = engineer_weather_features(df, DATA_FOLDER / WEATHER)
    return df


def get_Xy(basetable: pd.DataFrame, target: str = "max"):
    X = basetable.drop(columns=["value_max", "value_min"]).values
    if target == "max":
        y = basetable.value_max.values
    elif target == "min":
        y = basetable.value_min.values
    else:
        raise ValueError
    return X, y


basetable = combine_features_targets(TRAINING_FEATURES, TRAINING_TARGETS)
basetable = engineer_features(basetable)


# max values
clf_max = RandomForestRegressor(n_estimators=12)
X_max, y_max = get_Xy(basetable, target="max")
clf_max = clf_max.fit(X_max, y_max)
clf_min = RandomForestRegressor(n_estimators=12)
X_min, y_min = get_Xy(basetable, target="min")
clf_min = clf_min.fit(X_min, y_min)

august_features = pd.read_csv(
    DATA_FOLDER / AUGUST_FEATURES,
    parse_dates=["time"],
    index_col="time",
    dayfirst=True,
)
august_features = engineer_features(august_features)


X_august = august_features

predictions = pd.DataFrame(
    {
        "value_max": clf_max.predict(X_august),
        "value_min": clf_min.predict(X_august),
    },
    index=X_august.index,
)

predictions.to_csv("predictions.csv")


# TODO: look into permutation feature importance

# clf.predict()


# plt.plot(y[:48], label="Target")
# plt.plot(X[:48, 0], ls="--", label="Mean value")
# plt.plot(clf.predict(X[:48]), label="Prediction")
# plt.legend()
# plt.show()

# # TODO: test on validation set, look at RMSE

# np.sqrt(mean_squared_error(y, clf.predict(X)))

# looks ok for the day in question - now apply to test data and extend to min

# training_set, validation_set = train_test_split(
#     basetable,
#     test_size=0.2,
#     random_state=7,
# )

# X_training, y_training = get_Xy(training_set, target='max')
# X_validation, y_validation = get_Xy(validation_set, target='max')

# results = pd.DataFrame(columns=["n_estimators", "RMSE_train", "RMSE_validation"])
# for n_estimators in range(1, 31):
#     """Get RMSE to find best n_estimators hyperparameter"""
#     clf = RandomForestRegressor(n_estimators=n_estimators)
#     clf = clf.fit(X_training, y_training)
#     RMSE_train = np.sqrt(mean_squared_error(y_training, clf.predict(X_training)))
#     RMSE_validation = np.sqrt(
#         mean_squared_error(y_validation, clf.predict(X_validation))
#     )
#     results = results.append(
#         {
#             "n_estimators": n_estimators,
#             "RMSE_train": RMSE_train,
#             "RMSE_validation": RMSE_validation,
#         },
#         ignore_index=True,
#     )

# results.set_index("n_estimators").plot()
