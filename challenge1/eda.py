from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

DATA_FOLDER = Path("data")
MINUTES = "MW_Staplegrove_CB905_MW_minute_real_power_MW_pre_august.csv"
TARGETS = "MW_Staplegrove_CB905_MW_target_variable_half_hourly_max_min_real_power_MW_pre_august.csv"
OBSERVATIONS = "MW_Staplegrove_CB905_MW_observation_variable_half_hourly_real_power_MW_pre_august.csv"

minutes = pd.read_csv(
    DATA_FOLDER / MINUTES,
    parse_dates=["time"],
    usecols=["time", "value"],
    index_col="time",
    dayfirst=True,
)

# pick out some features from the datetime
attributes = ["year", "month", "dayofweek", "hour"]

for attr in attributes:
    if attr == "hour":
        minutes[attr] = getattr(minutes.index, attr) + minutes.index.minute / 60
    else:
        minutes[attr] = getattr(minutes.index, attr)

minutes["halfhour"] = minutes.index.floor("30min")

targets = pd.read_csv(
    DATA_FOLDER / TARGETS,
    parse_dates=["time"],
    index_col="time",
    dayfirst=True,
)

observations = pd.read_csv(
    DATA_FOLDER / OBSERVATIONS,
    parse_dates=["time"],
    index_col="time",
    dayfirst=True,
)

observations.rename(columns={"value": "value_halfhour"}, inplace=True)

minutes = minutes.merge(targets, left_on="halfhour", right_index=True)
minutes = minutes.merge(observations, left_on="halfhour", right_index=True)

minutes.drop(columns="halfhour", inplace=True)

# plot it

selected_day = pd.Timestamp(2020, 6, 15)
mask = minutes.index.date == selected_day

df_plot = minutes[
    [
        "value_max",
        "value_min",
        "value",
        "value_halfhour",
    ]
].copy()[mask]

sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 12, "axes.titlesize": 12})
sns.despine(offset=0, trim=True)

fig, axes = plt.subplots(
    figsize=(12, 6),
    nrows=2,
    sharex=True,
    gridspec_kw={"height_ratios": [2, 1]},
)

axes[0].plot(
    df_plot.value,
    alpha=0.8,
    color="black",
    lw=1,
    label="Minute observations",
)

axes[0].fill_between(
    df_plot.index,
    df_plot.value_min,
    df_plot.value_max,
    alpha=0.1,
    color="black",
    label="30min range",
)

axes[0].plot(
    df_plot.value_max,
    alpha=1,
    color="#F8766D",
    lw=1,
    label="30min max",
)

axes[0].plot(
    df_plot.value_min,
    alpha=1,
    color="#00C19F",
    lw=1,
    label="30min min",
)

axes[0].plot(
    df_plot.value_halfhour,
    alpha=1,
    color="#619CFF",
    lw=2,
    label="30min observations",
)

axes[0].xaxis.grid(False)
axes[0].legend()
axes[0].set_title(
    f"Real power (MW) by hour on {selected_day.strftime('%A %d %B %Y')}",
    loc="left",
)

axes[1].plot(
    df_plot.value_max - df_plot.value_min,
    color="#00B9E3",
    label="30min range",
)
axes[1].xaxis.grid(False)
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%H %M"))
axes[1].legend()
axes[1].margins(x=0)
axes[1].set_xlabel("Hour")
# axes[1].set_xlim(
#     pd.Timestamp(year=2020, month=6, day=15, hour=10),
#     pd.Timestamp(year=2020, month=6, day=15, hour=12),
# )

fig.tight_layout()


from matplotlib.animation import FuncAnimation
