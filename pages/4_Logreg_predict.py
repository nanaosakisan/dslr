import pandas as pd
from os.path import exists
import io
import streamlit as st

import settings


def logreg_predict(filename: str) -> None:
    if not exists(filename):
        st.error("Thetas file doesn't exist. Please run Logreg train.")
        return 

    thetas = pd.read_csv(filename, sep=";")
    filename = st.file_uploader("Fichier de test :", type="csv")
    if not filename:
        st.error("Please upload a file test.")
        return

    st.markdown("### Thetas :")
    st.write(thetas)
    data = pd.read_csv(io.StringIO(filename.read().decode('utf-8')), delimiter=',', index_col=0)
    data["Hogwarts House"] = data["Hogwarts House"].astype(str)
    data_schema = pd.DataFrame(pd.io.json.build_table_schema(data).get("fields"))
    true_schema = pd.read_json("./schema_train.json")
    if data_schema.equals(true_schema) :
        st.markdown("### Dataset")
        st.dataframe(data)
    else:
        st.write("Error in test dataframe.")

    
st.title("Logreg predict")
logreg_predict("./thetas.csv")