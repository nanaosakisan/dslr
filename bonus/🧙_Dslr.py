import streamlit as st
import pandas as pd
import io

import utils.settings as settings

st.set_page_config(page_title="DSLR", page_icon="🪄", layout="wide")

st.title("DSLR")
settings.init()
filename = st.file_uploader("Fichier d'entainement :", type="csv")

if filename:
    data = pd.read_csv(
        io.StringIO(filename.read().decode("utf-8")), delimiter=",", index_col=0
    )

    st.markdown("## Dataset")
    st.dataframe(data)
    settings.dataset = data
