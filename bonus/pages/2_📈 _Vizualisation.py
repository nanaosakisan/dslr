import plotly.express as px
import pandas as pd
import streamlit as st
import utils.settings as settings


def histogram(dataset: pd.DataFrame) -> None:
    st.markdown("## Histogram")
    name_col = dataset.columns.to_list()
    features = st.multiselect("Select features histogram", name_col)
    color_feature = st.selectbox(
        "Color feature histogram", [x for x in name_col if x not in features], index=0
    )
    if st.checkbox("Voir les histogrammes"):
        for f in features:
            name = px.histogram(
                dataset[[color_feature, f]],
                marginal="violin",
                title=f,
                color=color_feature,
            )
            st.write(name)


def scatter_plot(dataset: pd.DataFrame) -> None:
    name = dataset.columns
    st.markdown("## Scatter plot")
    feature1 = st.selectbox("Feature 1:", name)
    feature2 = st.selectbox("Feature 2:", name)
    if st.checkbox("Voir les scatter plot"):
        fig = px.scatter(
            dataset,
            x=feature1,
            y=feature2,
        )
        st.write(fig)


def pair_plot(dataset: pd.DataFrame) -> None:
    st.markdown("## Pair plot")
    name_col = data.columns.to_list()
    features = st.multiselect("Select features pair plot", name_col)
    color_feature = st.selectbox(
        "Color feature pair plot", [x for x in name_col if x not in features], index=0
    )
    if st.checkbox("Voir les pair plots"):
        fig_pair = px.scatter_matrix(dataset, dimensions=features, color=color_feature)
        fig_pair.update_traces(diagonal_visible=False, showupperhalf=False)
        st.write(fig_pair)


def vizualisation(dataset: pd.DataFrame) -> None:
    histogram(dataset)
    scatter_plot(dataset)
    pair_plot(dataset)


st.title("Vizualisation")
data = settings.dataset
if data.size == 0:
    st.info("Please upload a file.")
else:
    vizualisation(data)
