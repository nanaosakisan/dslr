import streamlit as st
import pandas as pd
import io

import utils.settings as settings

st.set_page_config(page_title="DSLR", page_icon="🪄", layout="wide")

st.title("DSLR")
settings.init()
filename = st.file_uploader("Fichier d'entainement :", type="csv")


def read_files(filename) -> pd.DataFrame:
    try:
        data = pd.read_csv(
            io.StringIO(filename.read().decode("utf-8")), delimiter=",", index_col=0
        )
    except Exception as e:
        st.error(f"{e.__class__} while reading dataset, please upload a valid file.")
        return pd.DataFrame()
    return data


if filename:
    data = read_files(filename)
    if data.size != 0:
        st.markdown("## Dataset")
        st.dataframe(data)
        settings.dataset = data
