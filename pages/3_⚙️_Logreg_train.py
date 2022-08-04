import streamlit as st
import numpy as np
import pandas as pd
import time
from typing import Tuple, List
import plotly.graph_objects as go

import utils.settings as settings
from utils.my_logistic_regression import MyLogisticRegression as MyLogReg
from utils.MinMaxNormalisation import MinMaxNormalisation
from utils.data_spliter import data_spliter
from utils.metrics import f1_score_


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if "log_time" in kw:
            name = kw.get("log_name", method.__name__.upper())
            kw["log_time"][name] = int((te - ts) * 1000)
        else:
            print("%r  %2.2f ms" % (method.__name__, (te - ts) * 1000))
        return result

    return timed


def get_features(dataset: pd.DataFrame) -> List[str]:
    col_names = dataset.columns
    nb_features = st.number_input(
        "Nb of features", min_value=2, max_value=len(col_names), value=2
    )

    features = []
    for i in range(nb_features):
        feature_name = st.selectbox("Feature " + str(i), col_names)
        features.append(feature_name)
    return features


def preprocessing_(
    dataset: pd.DataFrame, features: List[str], predict_value: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    select_col = features.copy()
    select_col.insert(0, predict_value)
    dataset = dataset[select_col].dropna()
    st.dataframe(dataset)
    dataset = dataset.to_numpy()

    X = np.array(dataset[:, 1:], dtype=float)
    Y = dataset[:, 0]
    encodage, Y_encode = np.unique(Y, return_inverse=True)
    settings.encodage = encodage
    X_train, X_test, Y_train, Y_test = data_spliter(
        X, Y_encode.reshape(-1, 1), proportion=0.8
    )

    scaler = MinMaxNormalisation()
    X_train_scale = scaler.fit_transform(X_train)
    X_test_scale = scaler.transform(X_test)
    return X_train_scale, X_test_scale, Y_train, Y_test


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
        for iter in range(5):
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


def display_f1(lr: List[pd.DataFrame]) -> None:
    fig = go.Figure()
    for c, lr in enumerate(lr):
        fig.add_trace(
            go.Scatter(
                x=lr["iter"],
                y=lr["f1_score"],
                name="f1_score " + str(settings.encodage[c]),
            )
        )
        fig.update_xaxes(title="Number of iteration")
        fig.update_yaxes(title="F1 score")
    st.write(fig)


def save_theta(best_model: pd.DataFrame) -> None:
    save = []
    for l in best_model:
        thetas = list(l["lr"].thetas.flatten())
        save.append(thetas)
    pd.DataFrame(save).to_csv("./thetas.csv", sep=";")


@timeit
def logreg_train(dataset: pd.DataFrame, **kwargs):
    name = dataset.columns
    st.markdown(
        "The two best features to select are : **Herbology** and **Defense against the dark arts**."
    )
    features = get_features(data)
    predict_value = st.selectbox("Value to predict:", name, index=0)
    validate_button = st.button("Validate")
    st.write(features)
    if validate_button == False:
        st.info("Please select features and click the validate button.")
    else:
        st.markdown("### Train")
        X_train, X_test, Y_train, Y_test = preprocessing_(
            dataset, features, predict_value
        )
        lr = fit_(X_train, X_test, Y_train, Y_test)
        best_model = select_best(lr)

        st.markdown("### Best models")
        col1, col2 = st.columns([2, 1])
        for c, best in enumerate(best_model):
            if c % 2 == 0:
                col1.markdown("**Model " + str(settings.encodage[c]) + "**")
                col1.write("F1_score : " + str(best["f1_score"]))
                col1.write("Thetas : ")
                col1.write(best["lr"].thetas)
            else:
                col2.markdown("**Model " + str(settings.encodage[c]) + "**")
                col2.write("F1_score : " + str(best["f1_score"]))
                col2.write("Thetas : ")
                col2.write(best["lr"].thetas)

        st.markdown("### Evaluation")
        display_f1(lr)
        save_theta(best_model)


st.title("Logreg train")
data = settings.dataset
if data.size == 0:
    st.info("Please upload a file.")
else:
    logreg_train(data)
