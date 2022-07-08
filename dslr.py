import streamlit as st

from utils.parse import parse_train
from describe import describe
from vizualisation import vizualisation
from logreg_train import logreg_train
from logreg_predict import logreg_predict

st.title("DSLR")

filename = st.file_uploader("Fichier d'entainement :")

if filename:
    res, error, dataset = parse_train(filename)
    if res == 1:
        st.markdown("## Dataset")
        st.dataframe(dataset)
        st.markdown("## Describe")
        des = describe(dataset)
        st.dataframe(des)
        vizualisation(dataset)

        logreg_train(dataset)
        logreg_predict("./thetas.csv")
    else:
        st.write(error)
