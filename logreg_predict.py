import pandas as pd
from os.path import exists
import streamlit as st


def logreg_predict(filename: str) -> None:
    if exists(filename):
        thetas = pd.read_csv(filename)
        st.write(thetas)
