from datetime import datetime

import kedro_light as kl
import plotly.express as px
import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq, fftshift


MINUTES_PER_DAY = 60 * 24
MINUTES_PER_INTERVAL = 30


def prepare(df):
    df["time"] = pd.to_datetime(df["time"])
    df["date"] = df["time"].apply(datetime.date)
    df["time"] = df["time"].apply(datetime.time)
    df["interval"] = [str((j % MINUTES_PER_DAY) // MINUTES_PER_INTERVAL) for j in df.index]
    return df


def plot_timeseries(minute):
    animation_frame = minute["date"].apply(str)  # plotly bug: can't animate on datetime
    fig = px.line(
        data_frame=minute,
        x="time",
        y="value",
        color="interval",
        animation_frame=animation_frame,
        range_x=(minute["time"].min(), minute["time"].max()),
        range_y=(minute["value"].min(), minute["value"].max()),
    )
    return fig.show()


# spectrum shows no clear cyclic patterns less than a single day
def plot_spectrum(df):
    df = df.iloc[:MINUTES_PER_DAY]  # one day
    n = len(df)
    dt = 1 / MINUTES_PER_DAY  # days per sample
    amp = np.abs(fftshift(fft(df["value"].values)))
    freq = fftshift(fftfreq(n, dt))
    px.line(x=freq, y=amp, log_y=True, labels={"x": "Cycles per day", "y": "Amplitude"}).show()


io = kl.io(conf_paths="conf", catalog="catalog.yml")
dag = [
    kl.node(func=prepare, inputs="raw_highres_xy", outputs="prep_highres_xy"),
    kl.node(func=prepare, inputs="raw_train_x", outputs="prep_train_x"),
    kl.node(func=prepare, inputs="raw_train_y", outputs="prep_train_y"),
    kl.node(func=plot_timeseries, inputs="prep_highres_xy", outputs=None),
    kl.node(func=plot_spectrum, inputs="prep_highres_xy", outputs=None),
]
kl.run(dag, io)
