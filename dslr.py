import streamlit as st
import pandas as pd
import io

from utils.parse import parse_train
from describe import describe
from vizualisation import vizualisation
from logreg_train import logreg_train
from logreg_predict import logreg_predict

st.title("DSLR")

filename = st.file_uploader("Fichier d'entainement :")

if filename:
    data = pd.read_csv(io.StringIO(filename.read().decode('utf-8')), delimiter=',', index_col=0)
    data_schema = pd.DataFrame(pd.io.json.build_table_schema(data).get("fields"))
    true_schema = pd.read_json("./schema.json")
    if data_schema.equals(true_schema) :
        st.markdown("## Dataset")
        st.dataframe(data)
        st.markdown("## Describe")
        des = describe(data)
        st.dataframe(des.astype(str))
        vizualisation(data)

    #     # logreg_train(dataset)
    #     # logreg_predict("./thetas.csv")
    else:
        st.write("Error in train dataframe")
