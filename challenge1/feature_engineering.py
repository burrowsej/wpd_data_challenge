from pathlib import Path
import numpy as np
import pandas as pd
from findiff import FinDiff
import calendar
from typing import Optional, Dict
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    PolynomialFeatures,
    MinMaxScaler,
    Normalizer,
    StandardScaler,
)
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_squared_error


def read_data(path: Path) -> pd.DataFrame:
    """Read a csv into a pandas dataframe"""
    df = pd.read_csv(
        path,
        parse_dates=["time"],
        index_col="time",
        dayfirst=True,
    )
    return df


def combine_features_targets(features: Path, targets: Path) -> pd.DataFrame:
    """Read in features and targets and combine in a single dataframe"""
    features = read_data(features)
    targets = read_data(targets)
    features = features.merge(targets, left_index=True, right_index=True)
    return features


class EngineerTemporalFeatures(BaseEstimator, TransformerMixin):
    """Extracts temporal features from a datetime index"""

    def __init__(
        self,
        cyclical_encoding: bool = True,
        binary_weekend: bool = False,
        include_year: bool = False,
    ):
        self.cyclical_encoding = cyclical_encoding
        self.binary_weekend = binary_weekend
        self.include_year = include_year

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df["annual_quantile"] = df.index.map(
            lambda x: x.dayofyear / (366 if calendar.isleap(x.year) else 365)
        )
        df["hour"] = df.index.hour + df.index.minute / 60

        # weekend binary feature performed worse
        if self.binary_weekend:
            df["weekend"] = df.index.dayofweek.map(lambda x: True if x > 4 else False)
        else:
            df["weekly_quantile"] = df.index.dayofweek

        if self.include_year:
            df["year"] = df.index.year

        if self.cyclical_encoding:
            # encode cyclical features in two separate sin and cos transforms
            max_vals = {"weekly_quantile": 6, "hour": 23.5, "annual_quantile": 366}
            for col in max_vals:
                df[f"sin_{col}"] = np.sin((2 * np.pi * df[col]) / max_vals[col])
                df[f"cos_{col}"] = np.cos((2 * np.pi * df[col]) / max_vals[col])
                # df.plot.scatter(f"sin_{col}", f"cos_{col}").set_aspect("equal")
                df.drop(columns=col, inplace=True)
        return df


class EngineerDemandFeatures(BaseEstimator, TransformerMixin):
    """Extracts demand features from 30 min values"""

    def __init__(
        self,
        include_finite_diff: bool = True,
        finite_diff_accuracy: int = 2,
        finite_diff_depth: int = 6,
        include_noise_feature: bool = True,
        noise_polynomial_order: int = 7,
        include_basic_forward_backward_diff: bool = True,
    ):
        self.include_finite_diff = include_finite_diff
        self.finite_diff_accuracy = finite_diff_accuracy
        self.finite_diff_depth = finite_diff_depth
        self.include_noise_feature = include_noise_feature
        self.noise_polynomial_order = noise_polynomial_order
        self.include_basic_forward_backward_diff = False

    def get_rmse(self, day: pd.DataFrame) -> float:
        """Fit a polynomial and get the rmse to quantify noise for the day"""
        X = np.c_[day.index.hour + day.index.minute / 60]
        y = day.value.values
        polyreg = make_pipeline(
            PolynomialFeatures(degree=self.noise_polynomial_order),
            LinearRegression(),
        )
        polyreg.fit(X, y)
        return np.sqrt(mean_squared_error(y, polyreg.predict(X)))

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # finite difference method to differentiate the values by different
        # orders - requires findiff package
        if self.include_finite_diff:
            for d in range(1, self.finite_diff_depth + 1):
                derivative = FinDiff(0, self.finite_diff_accuracy, d)
                df[f"d{d}_value"] = derivative(df.value)

        # fit a polynomial and calculate the 'noise' as the RMSE
        if self.include_noise_feature:
            df["date"] = df.index.date
            rmse_by_day = df.groupby("date").apply(self.get_rmse)
            rmse_by_day = pd.Series(rmse_by_day, name="daily_noise")
            df.drop(columns="date", inplace=True)
            df = df.merge(rmse_by_day, left_index=True, right_index=True, how="left")
            df.daily_noise.ffill(inplace=True)

        # basic forward and backwards difference
        if self.include_basic_forward_backward_diff:
            df["diff_forwards"] = df.value.diff(periods=-1).ffill()
            df["diff_backwards"] = df.value.diff(periods=1).bfill()

        return df


class EngineerWeatherFeatures(BaseEstimator, TransformerMixin):
    """Loads weather data from weather_path, upsamples to 30mins and engineers
    some of the features"""

    def __init__(
        self,
        weather_path: Path,
        adjust_15mins: Optional[str] = None,  # also valid are 'forward', 'backward'
        interpolation_kwargs: Dict = dict(method="spline", order=3),
        include_finite_diff_irradiance: bool = True,
        finite_diff_accuracy: int = 2,
        finite_diff_depth: int = 6,
    ):
        self.weather_path = weather_path
        self.adjust_15mins = adjust_15mins
        self.interpolation_kwargs = interpolation_kwargs
        self.include_finite_diff_irradiance = include_finite_diff_irradiance
        self.finite_diff_accuracy = finite_diff_accuracy
        self.finite_diff_depth = finite_diff_depth

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        weather = pd.read_csv(
            self.weather_path,
            parse_dates=["datetime"],
            index_col="datetime",
            dayfirst=True,
        )

        # resample to 30mins to get same as demand data
        weather = weather.resample("30T").interpolate(**self.interpolation_kwargs)

        # vectorise wind speed components
        weather["windspeed"] = weather.apply(
            lambda x: np.sqrt(x.windspeed_north ** 2 + x.windspeed_east ** 2), axis=1
        )
        weather["winddirection"] = weather.apply(
            lambda x: np.arctan2(x.windspeed_north, x.windspeed_east), axis=1
        )

        # cyclical encoding of wind direction
        # TODO: read up on this and explore - should not need the pis and the 2s
        weather["sin_winddirection"] = np.sin(
            (2 * np.pi * weather["winddirection"]) / np.pi
        )
        weather["cos_winddirection"] = np.cos(
            (2 * np.pi * weather["winddirection"]) / np.pi
        )
        # weather.plot.scatter("sin_winddirection", "cos_winddirection").set_aspect("equal")
        weather.drop(columns="winddirection", inplace=True)

        # adjust data forward or backward 15mins
        if self.adjust_15mins == "forward":
            weather = weather.resample("15min").interpolate()
            weather = weather.iloc[1::2, :]
            weather.index = weather.index - pd.Timedelta("15min")
        elif self.adjust_15mins == "backward":
            weather = weather.resample("15min").interpolate()
            weather = weather.iloc[3::2, :]
            weather.index = weather.index + pd.Timedelta("15min")

        # finite difference method to differentiate the irradiance by different
        # orders - requires findiff package
        if self.include_finite_diff_irradiance:
            for d in range(1, self.finite_diff_depth + 1):
                derivative = FinDiff(0, self.finite_diff_accuracy, d)
                weather[f"d{d}_solar_irradiance"] = derivative(weather.solar_irradiance)

        df = df.merge(weather, left_index=True, right_index=True, how="left")

        return df
