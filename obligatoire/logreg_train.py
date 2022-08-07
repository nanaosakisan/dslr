import argparse
import numpy as np
import pandas as pd
from typing import Tuple, List

from utils.my_logistic_regression import MyLogisticRegression as MyLogReg
from utils.MinMaxNormalisation import MinMaxNormalisation
from utils.data_spliter import data_spliter
from utils.metrics import f1_score_


def preprocessing_(
    dataset: pd.DataFrame, features: List[str], predict_value: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    select_col = features.copy()
    select_col.insert(0, predict_value)
    dataset = dataset[select_col].dropna()
    print(f"Dataset before encodage\n{dataset}")
    dataset = dataset.to_numpy()

    X = np.array(dataset[:, 1:], dtype=float)
    Y = dataset[:, 0]
    encodage, Y_encode = np.unique(Y, return_inverse=True)
    encodage = encodage
    X_train, X_test, Y_train, Y_test = data_spliter(
        X, Y_encode.reshape(-1, 1), proportion=0.8
    )

    scaler = MinMaxNormalisation()
    X_train_scale = scaler.fit_transform(X_train)
    X_test_scale = scaler.transform(X_test)
    return X_train_scale, X_test_scale, Y_train, Y_test, encodage


def uni_train(
    X_train: np.ndarray,
    X_test: np.ndarray,
    Y_train: np.ndarray,
    Y_test: np.ndarray,
    feature: int,
) -> np.array:
    Y_train_feature = np.where(Y_train == feature, 1, 0).reshape(-1, 1)
    Y_test_feature = np.where(Y_test == feature, 1, 0).reshape(-1, 1)

    lr_trained = []
    for alpha_power in range(-1, -2, -1):
        alpha = 10**alpha_power
        thetas = np.zeros((X_train.shape[1] + 1, 1))
        lr = MyLogReg(thetas, alpha=alpha, max_iter=100)
        for iter in range(300):
            lr.fit_(X_train, Y_train_feature)
            pred = np.where(lr.predict_(X_test) > 0.5, 1, 0)
            f1_score = f1_score_(Y_test_feature, pred)
            lr_trained.append([lr, alpha, iter * 100, f1_score])
    return pd.DataFrame(lr_trained, columns=["lr", "alpha", "iter", "f1_score"])


def fit_(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    Y_train: pd.DataFrame,
    Y_test: pd.DataFrame,
) -> List[pd.DataFrame]:
    nb_predict = np.unique(Y_train).shape[0]
    lr = []
    for p in range(nb_predict):
        lr.append(uni_train(X_train, X_test, Y_train, Y_test, p))
    return lr


def select_best(lr: List[pd.DataFrame]) -> List[pd.DataFrame]:
    best = []
    for l in lr:
        best_tmp = l.iloc[l["f1_score"].idxmax()]
        best.append(best_tmp)
    return best


def save_theta(best_model: pd.DataFrame, encodage: pd.DataFrame) -> None:
    save = []
    for l in best_model:
        thetas = list(l["lr"].thetas.flatten())
        save.append(thetas)
    pd.DataFrame(save).to_csv("./obligatoire/thetas.csv", sep=";")
    pd.DataFrame(encodage).to_csv("./obligatoire/encodage.csv", sep=";")
    print("Save thetas and encodage : ok")


def logreg_train(dataset: pd.DataFrame):
    print("Train")
    X_train, X_test, Y_train, Y_test, encodage = preprocessing_(
        dataset, ["Herbology", "Defense Against the Dark Arts"], "Hogwarts House"
    )
    lr = fit_(X_train, X_test, Y_train, Y_test)
    best_model = select_best(lr)

    print("Best models")
    for c, best in enumerate(best_model):
        print(f"Model {str(encodage[c])}")
        print("F1_score : " + str(best["f1_score"]))
        print("Thetas : ")
        print(best["lr"].thetas)

    save_theta(best_model, encodage)


def read_files(args) -> pd.DataFrame:
    try:
        data = pd.read_csv(args.dataset, index_col=0)
    except Exception as e:
        print(f"{e.__class__} while reading dataset, please upload a valid file.")
        return pd.DataFrame()
    data_schema = pd.DataFrame(pd.io.json.build_table_schema(data).get("fields"))
    true_schema = pd.read_json("./obligatoire/utils/schema.json")
    if data_schema.equals(true_schema) != True:
        print("Schema error, please upload a valid file.")
        print(f"data_schema\n{data_schema}")
        print(f"true_schema\n{true_schema}")
        return pd.DataFrame()
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Train and select the best models to predict house."
    )
    parser.add_argument("dataset", type=str, help="Name of the train dataset")
    args = parser.parse_args()
    data = read_files(args)
    if data.size == 0:
        return
    print(f"Dataset\n{data}")
    logreg_train(data)


main()
