import pandas as pd
import numpy as np
from os.path import exists
import io
import streamlit as st
from typing import Optional

import utils.settings as settings
from utils.my_logistic_regression import MyLogisticRegression as MyLogReg


def predict(data: np.ndarray, thetas: np.ndarray) -> np.ndarray:
    logreg = []
    for t in thetas:
        logreg.append(MyLogReg(thetas=t.reshape(-1, 1)))

    y0 = np.array(logreg[0].predict_(data))
    y1 = logreg[1].predict_(data)
    y_pred = np.concatenate((y0, y1), axis=1)
    y2 = logreg[2].predict_(data)
    y_pred = np.concatenate((y_pred, y2), axis=1)
    y3 = logreg[3].predict_(data)
    y_pred = np.concatenate((y_pred, y3), axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    return y_pred


def get_thetas(filename: str) -> Optional[pd.DataFrame]:
    if not exists(filename):
        st.info("Thetas file doesn't exist. Please run Logreg train.")
        return
    thetas = pd.read_csv(filename, sep=";")
    return thetas


def get_dataset() -> pd.DataFrame:
    dataset_filename = st.file_uploader("Fichier de test :", type="csv")
    if not dataset_filename:
        st.info("Please upload a file test.")
        return

    data = pd.read_csv(
        io.StringIO(dataset_filename.read().decode("utf-8")), delimiter=",", index_col=0
    )
    # if type(data["Hogwarts House"][0]) == str:
    #     return pd.DataFrame()

    # data["Hogwarts House"] = data["Hogwarts House"].astype(str)
    # data_schema = pd.DataFrame(pd.io.json.build_table_schema(data).get("fields"))
    # true_schema = pd.read_json("./utils/schema.json")
    # if not data_schema.equals(true_schema):
    #     return pd.DataFrame
    return data


def prediction_(
    data: pd.DataFrame, thetas: pd.DataFrame, feature1: str, feature2: str
) -> np.ndarray:
    X = np.array(data[[feature1, feature2]])
    pred = predict(X, thetas.iloc[:, 1:].to_numpy())
    st.markdown("## Predict")
    encodage = settings.encodage
    if encodage.size == 0:
        st.info("Encodage setting doesn't exist. Please run Logreg train.")
        return
    pred_decode = encodage[pred]
    return pred_decode


def save_house(pred_decode: np.ndarray):
    save_pred = pd.DataFrame(pred_decode, columns=["Hogwarts House"])
    save_pred.index.name = "Index"
    save_pred.to_csv("./houses.csv", sep=",")


def logreg_predict(thetas_filename: str) -> None:
    data = get_dataset()
    thetas = get_thetas(thetas_filename)
    if data is None:
        return
    elif data.size == 0:
        st.error("Error in test dataframe. Please upload a valid dataset.")
        return

    st.markdown("#### Thetas")
    st.write(thetas)
    st.markdown("#### Dataset")
    st.dataframe(data)
    st.markdown("### Prediction")
    st.write(
        "The two best features to select are : **Herbology** and **Defense against the dark arts**."
    )

    name = data.columns
    feature1 = st.selectbox("Feature 1:", name)
    feature2 = st.selectbox("Feature 2:", name)
    validate_button = st.button("Validate")
    if feature1 == feature2 or validate_button == False:
        st.info("Please select differents features and click the validate button.")
        return

    pred_decode = prediction_(data, thetas, feature1, feature2)
    if pred_decode is None:
        return
    st.dataframe(pred_decode)
    save_house(pred_decode)


st.title("Logreg predict")
logreg_predict("./thetas.csv")
