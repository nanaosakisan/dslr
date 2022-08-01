from xmlrpc.client import boolean
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
    data_schema = pd.DataFrame(pd.io.json.build_table_schema(data).get("fields"))
    true_schema = pd.read_json("./utils/schema.json")

    if data_schema.equals(true_schema):
        st.markdown("## Dataset")
        st.dataframe(data)
        settings.dataset = data
    else:
        st.write("Error in train dataframe.")
