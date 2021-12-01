from pathlib import Path
import numpy as np
import pandas as pd
import calendar
from typing import Optional
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    PolynomialFeatures,
    MinMaxScaler,
    Normalizer,
    StandardScaler,
)
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    BaggingRegressor,
    AdaBoostRegressor,
    VotingRegressor,
    StackingRegressor,
)
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.inspection import permutation_importance
from sklearn.multioutput import MultiOutputRegressor
from sklearn.compose import TransformedTargetRegressor
from functools import lru_cache
from findiff import FinDiff
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from typing import List


TARGETS = ["value_max", "value_min"]
rng = np.random.RandomState(7)
# REGRESSOR = RandomForestRegressor(random_state=rng)  # 0.408
# REGRESSOR = GradientBoostingRegressor(random_state=rng)  # 0.428
# REGRESSOR = XGBRegressor(random_state=rng)  # 0.418
# REGRESSOR = BaggingRegressor(XGBRegressor(random_state=rng))  # 0.4008
# REGRESSOR = AdaBoostRegressor(
#     XGBRegressor(random_state=rng), random_state=rng
# )  # 0.3987, also was 0.3969
# REGRESSOR = HistGradientBoostingRegressor(random_state=rng)  # 0.409
# REGRESSOR = LGBMRegressor(random_state=rng)  # 0.410
# REGRESSOR = BaggingRegressor(LGBMRegressor(random_state=rng))  # 0.3981
# REGRESSOR = AdaBoostRegressor(LGBMRegressor(random_state=rng))  # 0.4101

REGRESSOR = VotingRegressor(
    [
        ("rf", RandomForestRegressor(random_state=rng)),
        ("xgb", XGBRegressor(random_state=rng)),
        ("lgbm", LGBMRegressor(random_state=rng)),
        # ("hgb", HistGradientBoostingRegressor(random_state=rng)),
    ]
)

# REGRESSOR = StackingRegressor(
#     [
#         ("rf", RandomForestRegressor(random_state=rng)),
#         ("xgb", XGBRegressor(random_state=rng)),
#         ("lgbm", LGBMRegressor(random_state=rng)),
#         ("hgb", HistGradientBoostingRegressor(random_state=rng)),
#     ],
#     RidgeCV(),
# )


def get_Xy(df, targets=TARGETS):
    """Splits dataframe into X features and ys targets"""
    features = [col for col in df.columns if col not in targets]
    X = df[features]
    ys = df[targets].values
    return X, ys


def transform_targets(regressor=REGRESSOR):
    """Transforms targets from max, min to mean and range"""

    forward = lambda ys: np.vstack(
        (
            # skew component (mean which can be compared to value)
            0.5 * ys.sum(axis=1),
            # dispersion component (range)
            ys[:, 1] - ys[:, 0],
        )
    ).T
    backward = lambda ys: np.vstack(
        (ys[:, 0] - 0.5 * ys[:, 1], ys[:, 0] + 0.5 * ys[:, 1])
    ).T

    # reg = make_pipeline(
    #     # MinMaxScaler(),
    #     # Normalizer(),
    #     MultiOutputRegressor(estimator=regressor),
    # )

    reg = MultiOutputRegressor(estimator=regressor)

    reg = TransformedTargetRegressor(reg, func=forward, inverse_func=backward)

    return reg


def train(X, ys, targets=TARGETS, reg=REGRESSOR):
    """Splits dataframe into X and ys, transforms the targets and trains a model"""
    # X, ys = get_Xy(df, targets)

    reg = transform_targets(reg)
    reg.fit(X, ys)
    return reg


def train_mean(df, targets=TARGETS, reg=REGRESSOR):
    """Splits dataframe into X and ys, transforms the targets and trains a model"""
    target_name = "target_mean"

    df2 = df.copy()
    df2[target_name] = df2[TARGETS].mean(axis=1)
    df2.drop(columns=TARGETS, inplace=True)

    # mean
    X, y = df2.drop(columns=target_name), df2[target_name].values

    reg.fit(X, y)
    return reg


def train_range(df, targets=TARGETS, reg=REGRESSOR):
    """Splits dataframe into X and ys, transforms the targets and trains a model"""
    target_name = "target_range"

    df2 = df.copy()
    df2[target_name] = df2[TARGETS].diff(axis=1).iloc[:, 1]
    df2.drop(columns=TARGETS, inplace=True)

    # mean
    X, y = df2.drop(columns=target_name), df2[target_name].values

    reg.fit(X, y)
    return reg


def gridsearch(
    df,
    targets=TARGETS,
    reg=REGRESSOR,
    parameters={
        "regressor__estimator__n_estimators": range(50, 300, 50),
        # "regressor__estimator__max_features": range(),
    },
):

    X, ys = get_Xy(df, targets)
    reg = transform_targets(reg)
    reg = GridSearchCV(reg, parameters)
    reg.fit(X, ys)

    return reg


def validate(reg, X, Y) -> pd.DataFrame:
    df = Y.copy()
    df[[f"pred({col})" for col in Y]] = reg.predict(X)
    df.rename(columns={col: f"true({col})" for col in Y}, inplace=True)
    return df


def get_rmse(df: pd.DataFrame, targets: List[str]) -> float:
    mse = 0
    for col in targets:
        mse += mean_squared_error(df[f"pred({col})"], df[f"true({col})"], squared=True)
    rmse = np.sqrt(mse)
    print(f"RMSE:{rmse:.4f}")
    return rmse


def benchmark(X: pd.DataFrame, Y: pd.DataFrame) -> pd.DataFrame:
    """benchmark assumes min=max=value"""
    df = Y.copy()
    df[[f"pred({col})" for col in Y]] = np.repeat(X["value"].values[:, None], 2, axis=1)
    df.rename(columns={col: f"true({col})" for col in Y}, inplace=True)
    return df


def get_score(score_model: float, score_benchmark: float) -> float:
    """Score is ratio of model RMSE and benchmark RMSE"""
    score = score_model / score_benchmark
    print(f"Score:{score:.4f}")
    return score


def get_feature_importance(df, reg=REGRESSOR) -> pd.DataFrame():
    """Return a dataframe of the importance of each feature and plot it"""

    X, ys = get_Xy(df)
    reg = transform_targets(reg)
    reg.fit(X, ys)

    r = permutation_importance(reg, X, ys)

    feature_names = df.drop(columns=TARGETS).columns.to_list()
    features_importance = pd.DataFrame(
        {
            "mean_importance": r.importances_mean,
            "std_importance": r.importances_std,
        },
        index=feature_names,
    )

    features_importance.sort_values(by="mean_importance", inplace=True)

    features_importance.mean_importance.plot.barh(
        xerr=features_importance.std_importance,
        logx=True,
        figsize=(10, 5),
        title="Feature importance",
    )

    return features_importance


# import dabl

# basetable2 = basetable.copy()
# basetable2["target_mean"] = basetable2[TARGETS].mean(axis=1)
# basetable2["target_range"] = basetable[TARGETS].diff(axis=1).iloc[:, 1]
# basetable2.drop(columns=TARGETS, inplace=True)


# dabl.plot(basetable2.drop(columns="target_mean"), "target_range")
# dabl.plot(basetable2.drop(columns="target_range"), "target_mean")
