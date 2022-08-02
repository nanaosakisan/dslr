from base64 import encode
import pandas as pd
import numpy as np

from utils.MinMaxNormalisation import MinMaxNormalisation


def init() -> None:
    global dataset
    global encodage
    global lr0
    global lr1
    global lr2
    global lr3

    dataset = pd.DataFrame()
    encodage = pd.DataFrame()
    lr0 = []
    lr1 = []
    lr2 = []
    lr3 = []
