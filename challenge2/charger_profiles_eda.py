import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv(
    "data/charging_profiles_sub_set.csv", parse_dates=["TimeStamp"], dayfirst=True
)

fig, ax = plt.subplots(1, 1, figsize=(12, 6))
sns.lineplot(x="TimeStamp", y="Power_W", hue="ChargerID", data=df, ax=ax)
plt.show()
