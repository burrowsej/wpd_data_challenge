from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

DATA_FOLDER = Path("data")
FEATURES = "MW_Staplegrove_CB905_MW_minute_real_power_MW_pre_august.csv"
TARGETS = "MW_Staplegrove_CB905_MW_target_variable_half_hourly_max_min_real_power_MW_pre_august.csv"


features = pd.read_csv(
    DATA_FOLDER / FEATURES,
    parse_dates=["time"],
    usecols=["time", "value"],
    index_col="time",
    dayfirst=True,
)

# pick out some features from the datetime
attributes = ["year", "month", "dayofweek", "hour"]

for attr in attributes:
    if attr == "hour":
        features[attr] = getattr(features.index, attr) + features.index.minute / 60
    else:
        features[attr] = getattr(features.index, attr)

features["target_index"] = features.index.floor("30min")

targets = pd.read_csv(
    DATA_FOLDER / TARGETS,
    parse_dates=["time"],
    index_col="time",
    dayfirst=True,
)

features = features.merge(targets, left_on="target_index", right_index=True)
features.drop(columns="target_index", inplace=True)

# plot a day for visual check
features[["value_max", "value_min", "value"]].head(24 * 60).plot(figsize=(16, 6))

X = features.copy().drop(columns=["value_min", "value_max"]).values
Y = features.copy().value_max.values

clf = RandomForestRegressor(n_estimators=10)
clf = clf.fit(X, Y)
