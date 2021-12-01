from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from pipeline import (
    combine_features_targets,
    engineer_temporal_features,
    get_rmse,
    engineer_30min_demand_features,
    engineer_weather_features,
    train,
    get_feature_importance,
)

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
TARGETS = ["value_max", "value_min"]

basetable = combine_features_targets(
    DATA_FOLDER / TRAINING_FEATURES,
    DATA_FOLDER / TRAINING_TARGETS,
)
basetable = engineer_temporal_features(basetable)
basetable = engineer_30min_demand_features(basetable)
basetable = engineer_weather_features(basetable, DATA_FOLDER / WEATHER)
reg = train(basetable, TARGETS)


X_august = pd.read_csv(
    DATA_FOLDER / AUGUST_FEATURES,
    parse_dates=["time"],
    index_col="time",
    dayfirst=True,
)
X_august = engineer_temporal_features(X_august)
X_august = engineer_30min_demand_features(X_august)
X_august = engineer_weather_features(X_august, DATA_FOLDER / WEATHER)

predictions = pd.DataFrame(
    reg.predict(X_august),
    columns=TARGETS,
    index=X_august.index,
)

predictions.to_csv("predictions.csv")


# max values
# clf_max = RandomForestRegressor(n_estimators=12)
# X_max, y_max = get_Xy(basetable, target="max")
# clf_max = clf_max.fit(X_max, y_max)
# clf_min = RandomForestRegressor(n_estimators=12)
# X_min, y_min = get_Xy(basetable, target="min")
# clf_min = clf_min.fit(X_min, y_min)


# features_importance = get_feature_importance(clf_max, X_max, y_max)


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
