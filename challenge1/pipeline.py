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
from typing import List, Union
from sklearn.decomposition import PCA


TARGETS = ["value_max", "value_min"]
rng = np.random.RandomState(7)
REGRESSOR = RandomForestRegressor(random_state=rng)  # 0.408
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

# REGRESSOR = VotingRegressor(
#     [
#         ("rf", RandomForestRegressor(random_state=rng)),
#         ("xgb", XGBRegressor(random_state=rng)),
#         ("lgbm", LGBMRegressor(random_state=rng)),
#         # ("hgb", HistGradientBoostingRegressor(random_state=rng)),
#     ]
# )

# REGRESSOR = StackingRegressor(
#     [
#         ("rf", RandomForestRegressor(random_state=rng)),
#         ("xgb", XGBRegressor(random_state=rng)),
#         ("lgbm", LGBMRegressor(random_state=rng)),
#         ("hgb", HistGradientBoostingRegressor(random_state=rng)),
#     ],
#     RidgeCV(),
# )


def requires_plotly(func):
    global go
    import plotly.graph_objects as go

    return func


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


def validate(reg, X, Y) -> pd.DataFrame:
    df = Y.copy()
    df[[f"pred({col})" for col in Y]] = reg.predict(X)
    df.rename(columns={col: f"true({col})" for col in Y}, inplace=True)
    return df


def get_rmse(df: pd.DataFrame) -> float:
    split = int(df.shape[1] / 2)
    rmse = np.sqrt(
        mean_squared_error(
            df.iloc[:, :split],
            df.iloc[:, split:],
            multioutput="raw_values",
        ).sum()
    )
    # print(f"RMSE: {rmse:.4f}")
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
    # print(f"Score: {score:.4f}")
    return score


def scorer(reg, X, Y):
    """Scorer for use in cross val etc."""
    results = validate(reg, X, Y)
    bench = benchmark(X, Y)

    rmse_model = get_rmse(results)
    rmse_benchmark = get_rmse(bench)
    score = get_score(rmse_model, rmse_benchmark)
    return score


@requires_plotly
def pca_explained_variance(X: Union[pd.DataFrame, np.ndarray], n=None):

    n = X.shape[1] if n == None else n
    pca = PCA(n_components=n, random_state=rng)
    pca.fit(X)
    pca_results = pd.Series(
        np.cumsum(pca.explained_variance_ratio_), index=range(1, n + 1)
    )

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(figsize=(12,6))
    # ax.plot(pca_results.index, pca_results.values)
    # ax.set_xlabel('PCA Dimensions'), ax.set_ylabel('Explained Variance')
    # ax.set_yscale('log')
    # plt.show()

    fig = go.Figure(data=go.Scatter(x=pca_results.index, y=pca_results.values))
    fig.update_layout(
        xaxis_title="PCA Dimensions",
        yaxis_title="Explained Variance",
        yaxis_tickformat="%",
        xaxis_dtick=1,
    )

    fig.show(renderer="browser")


# def get_feature_importance(df, reg=REGRESSOR) -> pd.DataFrame():
#     """Return a dataframe of the importance of each feature and plot it"""

#     X, ys = get_Xy(df)
#     reg = transform_targets(reg)
#     reg.fit(X, ys)

#     r = permutation_importance(reg, X, ys)

#     feature_names = df.drop(columns=TARGETS).columns.to_list()
#     features_importance = pd.DataFrame(
#         {
#             "mean_importance": r.importances_mean,
#             "std_importance": r.importances_std,
#         },
#         index=feature_names,
#     )

#     features_importance.sort_values(by="mean_importance", inplace=True)

#     features_importance.mean_importance.plot.barh(
#         xerr=features_importance.std_importance,
#         logx=True,
#         figsize=(10, 5),
#         title="Feature importance",
#     )

#     return features_importance


# import dabl

# basetable2 = basetable.copy()
# basetable2["target_mean"] = basetable2[TARGETS].mean(axis=1)
# basetable2["target_range"] = basetable[TARGETS].diff(axis=1).iloc[:, 1]
# basetable2.drop(columns=TARGETS, inplace=True)


# dabl.plot(basetable2.drop(columns="target_mean"), "target_range")
# dabl.plot(basetable2.drop(columns="target_range"), "target_mean")
