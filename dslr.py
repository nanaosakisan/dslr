import streamlit as st

from parse import parse_train
from describe import describe
from vizualisation import vizualisation
from logreg_train import logreg_train

st.title("DSLR")

filename = st.file_uploader("Fichier d'entainement :")

if filename :
    res, error, dataset = parse_train(filename)
    if res == 1:
        col1, col2 = st.beta_columns(2)   

        st.markdown("## Dataset")
        st.dataframe(dataset)
        st.markdown("## Describe")
        des = describe(dataset)
        st.dataframe(des)
        vizualisation(dataset)

        logreg_train(dataset)
    else:
        st.write(error)
        