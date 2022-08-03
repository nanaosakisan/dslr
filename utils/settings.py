from base64 import encode
import pandas as pd
import numpy as np

from utils.MinMaxNormalisation import MinMaxNormalisation


def init() -> None:
    global dataset
    global encodage
    global thetas
    global lr

    dataset = pd.DataFrame()
    encodage = pd.DataFrame()
    thetas = np.array([])
    lr = []
