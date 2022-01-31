from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor,
    ExtraTreesRegressor,
    VotingRegressor,
)
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from typing import List

from feature_engineering import (
    read_data,
    combine_features_targets,
    EngineerTemporalFeatures,
    EngineerDemandFeatures,
    EngineerWeatherFeatures,
)
from pipeline import (
    TARGETS,
    transform_targets,
    train,
    train_mean,
    train_range,
    get_feature_importance,
    validate,
    get_rmse,
    benchmark,
    get_score,
    scorer,
)

DATA_FOLDER = Path("data")
TRAINING_FEATURES = "MW_Staplegrove_CB905_MW_observation_variable_half_hourly_real_power_MW_pre_august.csv"
TRAINING_TARGETS = "MW_Staplegrove_CB905_MW_target_variable_half_hourly_max_min_real_power_MW_pre_august.csv"
VALIDATION_FEATURES = (
    "MW_Staplegrove_CB905_MW_observation_variable_half_hourly_real_power_MW_august.csv"
)
VALIDATION_TARGETS = "MW_Staplegrove_CB905_MW_target_variable_half_hourly_max_min_real_power_MW_august.csv"
AUGUST_TEMPLATE = "Submission_template_august.csv"
TEST_TEMPLATE = "Submisson_template_september.csv"
TEST_FEATURES = "MW_Staplegrove_CB905_MW_observation_variable_half_hourly_real_power_MW_september.csv"
TEST_TEMPLATE = "Submisson_template_september.csv"
WEATHER = "df_staplegrove_1_hourly.csv"
rng = np.random.RandomState(42)

X_train = read_data(DATA_FOLDER / TRAINING_FEATURES)
Y_train = read_data(DATA_FOLDER / TRAINING_TARGETS)
X_validate = read_data(DATA_FOLDER / VALIDATION_FEATURES)
Y_validate = read_data(DATA_FOLDER / VALIDATION_TARGETS)

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.neural_network import MLPRegressor

REGRESSOR = VotingRegressor(
    [
        ("rf", RandomForestRegressor(random_state=rng)),
        ("xgb", XGBRegressor(random_state=rng)),
        ("lgbm", LGBMRegressor(random_state=rng)),
        ("hgb", HistGradientBoostingRegressor(random_state=rng)),
        ("et", ExtraTreesRegressor(random_state=rng)),
    ]
)  # 0.4443

X_train = pd.concat((X_train, X_validate))
Y_train = pd.concat((Y_train, Y_validate))

reg = make_pipeline(
    EngineerTemporalFeatures(cyclical_encoding=False),  # cyclical_encoding=False
    EngineerDemandFeatures(),  # noise_polynomial_order=10
    EngineerWeatherFeatures(weather_path=DATA_FOLDER / WEATHER),
    # transform_targets(RandomForestRegressor(random_state=rng, max_depth=17, n_jobs=-1)),
    # transform_targets(LGBMRegressor(random_state=rng, num_leaves=30, n_jobs=-1)),
    # transform_targets(XGBRegressor(random_state=rng, max_depth=3, n_jobs=-1)),
    # transform_targets(HistGradientBoostingRegressor(random_state=rng)),
    # transform_targets(ExtraTreesRegressor(random_state=rng)),
    # StandardScaler(),
    transform_targets(RandomForestRegressor(random_state=rng, max_depth=17, n_jobs=-1)),
    # transform_targets(MLPRegressor(random_state=rng)),
    # transform_targets(REGRESSOR),
)


# X_train = reg.transform(X_train)
# X_validate = reg.transform(X_validate)

# reg.fit(X_train, Y_train)

# print(f"\nTraining score: {scorer(reg, X_train, Y_train):.4f}")
# print(f"\nTesting score : {scorer(reg, X_validate, Y_validate):.4f}")
# print(
#     f"\nXval score: {cross_val_score(reg, X_train, Y_train, scoring=scorer, cv=3).mean():.4f}"
# )


param_grid = {}

#
grid_search = GridSearchCV(
    estimator=reg,
    param_grid=param_grid,
    scoring=scorer,
    cv=3,
    n_jobs=-1,
)

grid_search.fit(X_train, Y_train)

cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(f"{mean_score:.4f}", params)

print("\nBest...")
print(grid_search.best_estimator_)


# corr_mat=X_train.corr(method='pearson')
# plt.figure(figsize=(20,10))
# sns.heatmap(corr_mat,vmax=1,square=True,annot=True,cmap='cubehelix')
