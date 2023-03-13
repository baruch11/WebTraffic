"""load input data."""
import os
import pandas as pd
import numpy as np


def load_data():
    """Return data input dataframe."""
    this_dir = os.path.dirname(os.path.realpath(__file__))
    datapath = "data/web-traffic-time-series-forecasting/train_2.csv.zip"
    traffic_ds = pd.read_csv(os.path.join(this_dir, "..", datapath))
    return traffic_ds.set_index("Page").fillna(0).astype(np.int32)
