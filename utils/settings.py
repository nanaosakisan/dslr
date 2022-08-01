from base64 import encode
import pandas as pd
import numpy as np

from utils.MinMaxNormalisation import MinMaxNormalisation


def init() -> None:
    global dataset
    global encodage
    global scaler

    dataset = pd.DataFrame()
    encodage = pd.DataFrame()
