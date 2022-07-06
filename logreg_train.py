import streamlit as st
import numpy as np
import time
from typing import Tuple

from my_logistic_regression import MyLogisticRegression as MyLogReg
from ZscoreScaler import ZscoreScaler
from data_spliter import data_spliter
from metrics import f1_score_


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
    X: np.ndarray, Y_encode: np.ndarray, feature: int
) -> Tuple[MyLogReg, float]:
    Y_feature = np.where(Y_encode == feature, 1, 0).reshape(-1, 1)
    X_train, X_test, Y_train, Y_test = data_spliter(X, Y_feature, proportion=0.8)
    scaler = ZscoreScaler()
    x_train_scale = scaler.fit_transform(X_train)
    x_test_scale = scaler.transform(X_test)
    lr = MyLogReg(np.zeros(x_train_scale.shape(1) + 1, 1), alpha=1e-3, max_iter=20000)
    lr.fit_(x_train_scale, Y_train)
    f1_score = f1_score_(Y_test, lr.predict_(x_test_scale))
    return lr, f1_score


@timeit
def logreg_train(dataset: list, **kwargs):
    dataset = np.array(dataset)
    dataset = dataset[1:, [1, 2 + 5, 3 + 5]]
    dataset = dataset[~np.isnan(dataset)]

    print(type(dataset))

    st.markdown("## Entrainement")
    st.dataframe(dataset)
    X = dataset[:, [1, 2]]
    Y = dataset[:, 0]
    encodage, Y_encode = np.unique(Y, return_inverse=True)

    lr_0 = uni_train(X, Y_encode, 0)

    # ravenclaw = convert_one_vs_all(sub_data, 0)
    # st.write(ravenclaw)
    # slytherin = convert_one_vs_all(sub_data, 1)
    # gryffindor = convert_one_vs_all(sub_data, 2)
    # hufflepuff = convert_one_vs_all(sub_data, 3)

    st.datarame(lr_0.predict_(X))


# y_hat_test_normalized = np.where(y_hat_test < 0.5, 0, 1)
