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


def read_files(args) -> pd.DataFrame:
    try:
        data = pd.read_csv(args.dataset, index_col=0)
    except Exception as e:
        print(
            f"{e.__class__} while reading the dataset, please upload a valid dataset."
        )
        return pd.DataFrame()
    data_schema = pd.DataFrame(pd.io.json.build_table_schema(data).get("fields"))
    true_schema = pd.read_json("./obligatoire/utils/schema.json")
    if data_schema.equals(true_schema) != True:
        print("Schema error, please upload a valid file.")
        print(f"data_schema\n{data_schema}")
        print(f"true_schema\n{true_schema}")
        return pd.DataFrame()
    return data


def read_files(args) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    try:
        data = pd.read_csv(args.dataset, index_col=0)
    except Exception as e:
        print(f"{e.__class__} while reading dataset, please upload a valid file.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    try:
        thetas = pd.read_csv(args.thetas, sep=";")
    except Exception as e:
        print(f"{e.__class__} while reading thetas, please upload a valid file.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    try:
        encodage = pd.read_csv(args.encodage, sep=";")
    except Exception as e:
        print(f"{e.__class__} while reading encodage, please upload a valid file.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    return data, thetas, encodage


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
        description="Predict the house for student in the dataset"
    )
    parser.add_argument("dataset", type=str, help="Name of the test dataset")
    parser.add_argument("thetas", type=str, help="Name of the thetas file")
    parser.add_argument("--encodage", type=str, help="Name of the encodage file", default="obligatoire/encodage.csv")
    args = parser.parse_args()
    data, thetas, encodage = read_files(args)
    if data.size == 0 or thetas.size == 0 or encodage.size == 0:
        return
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
