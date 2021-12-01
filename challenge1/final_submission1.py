from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
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
    get_Xy,
    transform_targets,
    train,
    train_mean,
    train_range,
    gridsearch,
    get_feature_importance,
    validate,
    score,
    benchmark,
    skill,
)

DATA_FOLDER = Path("data")
TRAINING_FEATURES = "MW_Staplegrove_CB905_MW_observation_variable_half_hourly_real_power_MW_pre_august.csv"
TRAINING_TARGETS = "MW_Staplegrove_CB905_MW_target_variable_half_hourly_max_min_real_power_MW_pre_august.csv"
TEST_FEATURES = (
    "MW_Staplegrove_CB905_MW_observation_variable_half_hourly_real_power_MW_august.csv"
)
TEST_TARGETS = "MW_Staplegrove_CB905_MW_target_variable_half_hourly_max_min_real_power_MW_august.csv"
AUGUST_TEMPLATE = "Submission_template_august.csv"
SEPTEMBER_TEMPLATE = "Submisson_template_september.csv"
SEPTEMBER_FEATURES = "MW_Staplegrove_CB905_MW_observation_variable_half_hourly_real_power_MW_september.csv"
SEPTEMBER_TEMPLATE = "Submisson_template_september.csv"
WEATHER = "df_staplegrove_1_hourly.csv"
rng = np.random.RandomState(42)


basetable = combine_features_targets(
    DATA_FOLDER / TRAINING_FEATURES,
    DATA_FOLDER / TRAINING_TARGETS,
)


engineer_features = make_pipeline(
    EngineerTemporalFeatures(),
    EngineerDemandFeatures(),
    EngineerWeatherFeatures(weather_path=DATA_FOLDER / WEATHER),
    "passthrough",
)
reg = transform_targets(RandomForestRegressor(random_state=rng))


X_train = read_data(DATA_FOLDER / TRAINING_FEATURES)
Y_train = read_data(DATA_FOLDER / TRAINING_TARGETS)
X_test = read_data(DATA_FOLDER / TEST_FEATURES)
Y_test = read_data(DATA_FOLDER / TEST_TARGETS)

X_train = pd.concat((X_train, X_test))
Y_train = pd.concat((Y_train, Y_test))
X_train = engineer_features.transform(X_train)

reg.fit(X_train, Y_train)

results = validate(reg, X_train, Y_train)
benchmark = benchmark(X_train, Y_train)

score_model = score(results, Y_train.columns)
score_benchmark = score(benchmark, Y_train.columns)

skill(score_model, score_benchmark)


X_september = read_data(DATA_FOLDER / SEPTEMBER_FEATURES)
X_september = engineer_features.transform(X_september)

predictions = pd.DataFrame(
    reg.predict(X_september),
    columns=TARGETS,
    index=X_september.index,
)

# results.head(24*2).plot()

# template = pd.read_csv(DATA_FOLDER / SEPTEMBER_TEMPLATE)

predictions.to_csv("predictions.csv")


# print("Parameters currently in use:\n")

# print(reg.get_params())
