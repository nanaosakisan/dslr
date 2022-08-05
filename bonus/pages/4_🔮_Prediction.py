import pandas as pd
import numpy as np
from os.path import exists
import io
import streamlit as st
from typing import Optional, List

import utils.settings as settings
from utils.my_logistic_regression import MyLogisticRegression as MyLogReg


def read_files(filename) -> pd.DataFrame:
    try:
        data = pd.read_csv(
            io.StringIO(filename.read().decode("utf-8")), delimiter=",", index_col=0
        )
    except Exception as e:
        st.error(f"{e.__class__} while reading dataset, please upload a valid file.")
        return pd.DataFrame()
    return data


def predict(data: np.ndarray, thetas: np.ndarray) -> np.ndarray:
    logreg = []
    for t in thetas:
        logreg.append(MyLogReg(thetas=t.reshape(-1, 1)))

    y_pred = np.zeros((data.shape[0], 0), float)
    for l in logreg:
        y_tmp = l.predict_(data)
        y_pred = np.column_stack([y_pred, y_tmp])
    return np.argmax(y_pred, axis=1)


def get_features(dataset: pd.DataFrame, thetas: pd.DataFrame) -> List[str]:
    col_names = dataset.columns
    nb_features = thetas.shape[1] - 1
    features = []
    for i in range(nb_features):
        feature_name = st.selectbox("Feature " + str(i), col_names)
        features.append(feature_name)
    return features


def prediction_(
    data: pd.DataFrame,
    thetas: pd.DataFrame,
    encodage: pd.DataFrame,
    features: List[str],
) -> np.ndarray:
    X = data[features].to_numpy()
    pred = predict(X, thetas.to_numpy())
    st.markdown("## Predict")
    pred_decode = encodage[pred]
    return pred_decode


def logreg_predict() -> None:
    filename = st.file_uploader("Fichier de test :", type="csv")
    data = read_files(filename)
    if data.size == 0:
        return
    thetas = settings.thetas
    if thetas.size == 0:
        st.info("Error while getting thetas, please run Logreg train.")
        return
    encodage = settings.encodage
    if encodage.size == 0:
        st.info("Error while getting encodage, please run Logreg train.")
        return
    st.markdown("#### Dataset")
    st.dataframe(data)
    st.markdown("#### Thetas")
    st.write(thetas)
    st.markdown("### Prediction")
    st.write(
        "The two best features to select are : **Herbology** and **Defense against the dark arts**."
    )

    features = get_features(data, thetas)
    validate_button = st.button("Validate")
    if validate_button == False:
        st.info("Please select features and click the validate button.")
        return

    pred_decode = prediction_(data, thetas, features)
    st.dataframe(pred_decode)


st.title("Logreg predict")
logreg_predict()
