from base64 import encode
import pandas as pd

from utils.MinMaxNormalisation import MinMaxNormalisation


def init() -> None:
    global dataset
    global encodage
    global thetas
    global lr

    dataset = pd.DataFrame()
    encodage = pd.DataFrame()
    thetas = pd.DataFrame()
    lr = []
