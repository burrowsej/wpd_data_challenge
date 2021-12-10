from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestRegressor,
    VotingRegressor,
    ExtraTreesRegressor,
    StackingRegressor,
)
from sklearn.linear_model import RANSACRegressor, LinearRegression
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from typing import List
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


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
    validate,
    get_rmse,
    benchmark,
    get_score,
    scorer,
    pca_explained_variance,
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


basetable = combine_features_targets(
    DATA_FOLDER / TRAINING_FEATURES,
    DATA_FOLDER / TRAINING_TARGETS,
)


X_train = read_data(DATA_FOLDER / TRAINING_FEATURES)
Y_train = read_data(DATA_FOLDER / TRAINING_TARGETS)
X_validate = read_data(DATA_FOLDER / VALIDATION_FEATURES)
Y_validate = read_data(DATA_FOLDER / VALIDATION_TARGETS)

X_validate = X_validate[: len(Y_validate) - 27]
Y_validate = Y_validate[: len(Y_validate) - 27]

X_train = pd.concat((X_train, X_validate))
Y_train = pd.concat((Y_train, Y_validate))


# X_train = pipeline.transform(X_train)


# pca_explained_variance(X_train)

# pca = PCA(n_components=9, random_state=rng)
# X_train = pca.fit_transform(X_train)


# # print(pca.explained_variance_ratio_)

reg = make_pipeline(
    # PCA(n_components=2, random_state=rng),
    transform_targets(
        # RandomForestRegressor(random_state=rng, n_jobs=-1),  # 0.4550
        # ExtraTreesRegressor(random_state=rng, n_jobs=-1),
        # RANSACRegressor(
        #     RandomForestRegressor(random_state=rng, max_depth=18, n_jobs=-1),
        #     min_samples=29,
        #     random_state=rng,
        # ), # 0.47149
        # LGBMRegressor(random_state=rng, n_jobs=-1),  # 0.4506  # num_leaves=30,
        # XGBRegressor(random_state=rng, n_jobs=-1),  # 0.4643
        StackingRegressor(
            [
                ("et", ExtraTreesRegressor(random_state=rng)),
                ("rf", ExtraTreesRegressor(random_state=rng)),
                ("xgb", XGBRegressor(random_state=rng)),
                ("lgbm", LGBMRegressor(random_state=rng)),
                # ("hgb", HistGradientBoostingRegressor(random_state=rng)),
            ]
        ),
        # VotingRegressor(
        #     [
        # ("rf", RandomForestRegressor(random_state=rng)),
        #         ("rf", RandomForestRegressor(random_state=rng)),
        #         ("rf", ExtraTreesRegressor(random_state=rng)),
        #         ("xgb", XGBRegressor(random_state=rng)),
        #         ("lgbm", LGBMRegressor(random_state=rng)),
        #         # ("hgb", HistGradientBoostingRegressor(random_state=rng)),
        #     ]
        # ), #0.4332
    ),
)


pipeline = make_pipeline(
    EngineerTemporalFeatures(cyclical_encoding=True),
    EngineerDemandFeatures(),
    EngineerWeatherFeatures(weather_path=DATA_FOLDER / WEATHER),
    "passthrough",
)


X_train = pipeline.fit_transform(X_train)
X_validate = pipeline.fit_transform(X_validate)


# scores = cross_val_score(
#     reg,
#     X_train,  # .reset_index(drop=True)
#     Y_train.reset_index(drop=True),
#     scoring=scorer,
#     cv=5,
# )


# print(f"XvalScore: {scores.mean():.4f}")

# reg.fit(X_train, Y_train)

# scorer(
#     reg,
#     X_validate,
#     Y_validate,
# )


# param_grid = {
#     "regressor__estimator__max_depth": (6, 10, 14, 18, 22),
# }


# grid_search = GridSearchCV(
#     estimator=reg,
#     param_grid=param_grid,
#     scoring=scorer,
#     cv=5,
#     n_jobs=-1,
# )

# grid_search.fit(X_train.reset_index(drop=True), Y_train.reset_index(drop=True))

# cvres = grid_search.cv_results_
# for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
#     print(f"{mean_score:.4f}", params)

# print("\nBest...")
# print(grid_search.best_estimator_)

X_test = read_data(DATA_FOLDER / TEST_FEATURES)

X_test = pipeline.transform(X_test)

reg.fit(X_train, Y_train)


predictions = pd.DataFrame(
    reg.predict(X_test),
    columns=TARGETS,
    index=X_test.index,
)

# results.head(24*2).plot()

# template = pd.read_csv(DATA_FOLDER / SEPTEMBER_TEMPLATE)

predictions.to_csv("predictions.csv")


# print("Parameters currently in use:\n")

# print(reg.get_params())


# from sklearn.inspection import permutation_importance

# # # reg.fit(X_train, Y_train)

# result = permutation_importance(reg, X_train, Y_train)

# result.importances_mean


# feature_names = X_train.columns.to_list()
# features_importance = pd.DataFrame(
#     {
#         "mean_importance": result.importances_mean,
#         "std_importance": result.importances_std,
#     },
#     index=feature_names,
# )

# features_importance.sort_values(by="mean_importance", inplace=True)

# features_importance.mean_importance.plot.barh(
#     xerr=features_importance.std_importance,
#     logx=True,
#     figsize=(10, 5),
#     title="Feature importance",
# )
