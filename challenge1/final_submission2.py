from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import RANSACRegressor
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

X_train = pd.concat((X_train, X_validate))
Y_train = pd.concat((Y_train, Y_validate))

pipeline = make_pipeline(
    EngineerTemporalFeatures(cyclical_encoding=True),
    EngineerDemandFeatures(),
    EngineerWeatherFeatures(weather_path=DATA_FOLDER / WEATHER),
    "passthrough",
)

X_train = pipeline.transform(X_train)

reg = transform_targets(
    VotingRegressor(
        [
            ("rf", RandomForestRegressor(random_state=rng)),
            ("xgb", XGBRegressor(random_state=rng)),
            ("lgbm", LGBMRegressor(random_state=rng)),
        ]
    ),
)

# scores = cross_val_score(
#     reg,
#     X_train.reset_index(drop=True),
#     Y_train.reset_index(drop=True),
#     scoring=scorer,
#     cv=5,
# )

# print(f"XvalScore: {scores.mean():.4f}")


X_test = read_data(DATA_FOLDER / TEST_FEATURES)

X_test = pipeline.transform(X_test)

reg.fit(X_train.reset_index(drop=True), Y_train.reset_index(drop=True))


predictions = pd.DataFrame(
    reg.predict(X_test),
    columns=TARGETS,
    index=X_test.index,
)
predictions.to_csv("predictions.csv")
