import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from os.path import exists
import io
from typing import List, Tuple

from utils.my_logistic_regression import MyLogisticRegression as MyLogReg

FEATURES = [
    "Hogwarts House",
    "Arithmancy",
    "Astronomy",
    "Herbology",
    "Defense Against the Dark Arts",
    "Divination",
    "Muggle Studies",
    "Ancient Runes",
    "History of Magic",
    "Transfiguration",
    "Potions",
    "Care of Magical Creatures",
    "Charms",
    "Flying",
]


def check_features(name_col: list) -> bool:
    for f in FEATURES:
        if f not in name_col:
            return False
    return True


def read_files(args) -> Tuple[pd.DataFrame]:
    if Path(args.dataset).suffix != ".csv":
        print("Please upload a csv file as dataset.")
        return
    if Path(args.thetas).suffix != ".csv":
        print("Please upload a csv file as thetas.")
        return
    if Path(args.encodage).suffix != ".csv":
        print("Please upload a csv file as encodage.")
        return
    data = pd.read_csv(args.dataset)
    if check_features(data.columns.to_list()) == False:
        print(f"Please upload a valid dataset with at least features : {FEATURES}")
    thetas = pd.read_csv(args.thetas, sep=";")
    if thetas.size == 0:
        print("Please upload a valid thetas files")
    encodage = pd.read_csv(args.encodage, sep=";")
    if encodage.size == 0:
        print("Please upload a valid encodage files")
    return data, thetas, encodage


def predict(data: np.ndarray, thetas: np.ndarray) -> np.ndarray:

    return


def save_house(pred_decode: np.ndarray):
    save_pred = pd.DataFrame(pred_decode, columns=["Hogwarts House"])
    save_pred.index.name = "Index"
    save_pred.to_csv("./obligatoire/houses.csv", sep=",")


def prediction_(
    data: pd.DataFrame,
    thetas: pd.DataFrame,
    encodage: pd.DataFrame,
    features: List[str],
) -> None:
    X = data[features].to_numpy()
    logreg = []
    for t in thetas:
        logreg.append(MyLogReg(thetas=t.reshape(-1, 1)))

    y_pred = np.zeros((X.shape[0], 0), float)
    for l in logreg:
        y_tmp = l.predict_(X)
        y_pred = np.column_stack([y_pred, y_tmp])
    y_pred = np.argmax(y_pred, axis=1)
    print(encodage)
    pred_decode = encodage[y_pred]
    return pred_decode


def main():
    parser = argparse.ArgumentParser(
        description="Process some informations on the features of the dataset"
    )
    parser.add_argument("dataset", type=str, help="Name of the test dataset")
    parser.add_argument("thetas", type=str, help="Name of the thetas file")
    parser.add_argument("encodage", type=str, help="Name of the encodage file")
    args = parser.parse_args()
    data, thetas, encodage = read_files(args)
    print(f"Dataset\n{data}")
    print(f"Thetas\n{thetas}")
    pred_decode = prediction_(
        data,
        thetas.iloc[:, 1:].to_numpy(),
        encodage.iloc[:, 1:].to_numpy(),
        ["Herbology", "Defense Against the Dark Arts"],
    )
    if pred_decode is None:
        print("Prediction failed")
        return
    print(f"Prediction\n{pred_decode}")
    save_house(pred_decode)


main()
