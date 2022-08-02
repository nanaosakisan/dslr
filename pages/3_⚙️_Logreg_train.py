import streamlit as st
import numpy as np
import pandas as pd
import time
from typing import Tuple
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


def uni_train(
    X_train: np.ndarray,
    X_test: np.ndarray,
    Y_train: np.ndarray,
    Y_test: np.ndarray,
    feature: int,
) -> np.array:
    Y_train_feature = np.where(Y_train == feature, 1, 0).reshape(-1, 1)
    Y_test_feature = np.where(Y_test == feature, 1, 0).reshape(-1, 1)

    lr_trained = [["lr", "alpha", "iter", "f1_score"]]
    for alpha_power in range(-1, -2, -1):
        alpha = 10**alpha_power
        thetas = np.zeros((X_train.shape[1] + 1, 1))
        lr = MyLogReg(thetas, alpha=alpha, max_iter=100)
        for iter in range(500):
            lr.fit_(X_train, Y_train_feature)
            pred = np.where(lr.predict_(X_test) > 0.5, 1, 0)
            f1_score = f1_score_(Y_test_feature, pred)
            lr_trained.append([lr, alpha, iter * 100, f1_score])
    return np.array(lr_trained)


def preprocessing_(
    dataset: pd.DataFrame, feature1: str, feature2: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dataset = np.array(dataset[["Hogwarts House", feature1, feature2]].dropna())
    st.dataframe(dataset)

    X = np.array(dataset[:, [1, 2]], dtype=np.float)
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


def fit_(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    Y_train: pd.DataFrame,
    Y_test: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lr_0 = uni_train(X_train, X_test, Y_train, Y_test, 0)
    lr_1 = uni_train(X_train, X_test, Y_train, Y_test, 1)
    lr_2 = uni_train(X_train, X_test, Y_train, Y_test, 2)
    lr_3 = uni_train(X_train, X_test, Y_train, Y_test, 3)
    return lr_0, lr_1, lr_2, lr_3


def select_best(
    lr_0: np.ndarray, lr_1: np.ndarray, lr_2: np.ndarray, lr_3: np.ndarray
) -> Tuple[list, list, list, list]:
    best_0 = lr_0[np.where(lr_0 == np.max(lr_0[1:, 3]))[0][0].item()]
    best_1 = lr_1[np.where(lr_1 == np.max(lr_1[1:, 3]))[0][0].item()]
    best_2 = lr_2[np.where(lr_2 == np.max(lr_2[1:, 3]))[0][0].item()]
    best_3 = lr_3[np.where(lr_3 == np.max(lr_3[1:, 3]))[0][0].item()]
    return best_0, best_1, best_2, best_3


def save_theta(best_0: list, best_1: list, best_2: list, best_3: list) -> None:
    save = []
    theta_save_0 = list(best_0[0].thetas.flatten())
    theta_save_1 = list(best_1[0].thetas.flatten())
    theta_save_2 = list(best_2[0].thetas.flatten())
    theta_save_3 = list(best_3[0].thetas.flatten())
    save.append(theta_save_0)
    save.append(theta_save_1)
    save.append(theta_save_2)
    save.append(theta_save_3)
    pd.DataFrame(save).to_csv("./thetas.csv", sep=";")


def display_f1(
    lr0: np.ndarray, lr1: np.ndarray, lr2: np.ndarray, lr3: np.ndarray
) -> None:
    data0 = lr0[:, 1:]
    data0 = pd.DataFrame(data0, columns=["alpha", "iter", "f1"])
    data1 = lr1[:, 1:]
    data1 = pd.DataFrame(data1, columns=["alpha", "iter", "f1"])
    data2 = lr2[:, 1:]
    data2 = pd.DataFrame(data2, columns=["alpha", "iter", "f1"])
    data3 = lr3[:, 1:]
    data3 = pd.DataFrame(data3, columns=["alpha", "iter", "f1"])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=data0["iter"], y=data0["f1"], name="f1_score Gryffindor")
    )
    fig.add_trace(
        go.Scatter(x=data1["iter"], y=data1["f1"], name="f1_score Hufflepuff")
    )
    fig.add_trace(go.Scatter(x=data2["iter"], y=data2["f1"], name="f1_score Ravenclaw"))
    fig.add_trace(go.Scatter(x=data3["iter"], y=data3["f1"], name="f1_score Slytherin"))
    fig.update_layout(
        title="Evolution of the f1 score according to the number of iterations"
    )
    fig.update_xaxes(title="Number of iteration")
    fig.update_yaxes(title="F1 score")
    st.write(fig)


@timeit
def logreg_train(dataset: pd.DataFrame, **kwargs):
    name = dataset.columns[5:]
    st.markdown(
        "The two best features to select are : **Herbology** and **Defense against the dark arts**."
    )
    feature1 = st.selectbox("Feature 1:", name)
    feature2 = st.selectbox("Feature 2:", name)
    # stochiastic = st.checkbox("Stochiastic")
    validate_button = st.button("Validate")
    if feature1 == feature2 or validate_button == False:
        st.info("Please select differents features and click the validate button.")
    else:
        st.markdown("### Train")
        X_train, X_test, Y_train, Y_test = preprocessing_(dataset, feature1, feature2)
        lr0, lr1, lr2, lr3 = fit_(X_train, X_test, Y_train, Y_test)
        best_0, best_1, best_2, best_3 = select_best(lr0, lr1, lr2, lr3)

        st.markdown("### Best models")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.write("Model Gryffindor")
            st.write("F1_score: ", str(best_0[3]))
            st.write("Thetas: ", best_0[0].thetas)
        with col2:
            st.write("Model Hufflepuff")
            st.write("F1_score: ", str(best_1[3]))
            st.write("Thetas: ", best_1[0].thetas)
        with col3:
            st.write("Model Ravenclaw")
            st.write("F1_score: ", str(best_2[3]))
            st.write("Thetas: ", best_2[0].thetas)
        with col4:
            st.write("Model Slytherin")
            st.write("F1_score: ", str(best_3[3]))
            st.write("Thetas: ", best_3[0].thetas)

        st.markdown("### Evaluation")
        display_f1(lr0, lr1, lr2, lr3)

        save_theta(best_0, best_1, best_2, best_3)


st.title("Logreg train")
data = settings.dataset
if data.size == 0:
    st.error("Please upload a file.")
else:
    logreg_train(data)
