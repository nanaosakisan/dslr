import plotly.express as px
import streamlit as st

FEATURES = ["Arithmancy", "Astronomy","Herbology","Defense Against the Dark Arts","Divination","Muggle Studies","Ancient Runes","History of Magic","Transfiguration","Potions",\
    "Care of Magical Creatures","Charms","Flying"]

def histogram(dataset):
    st.markdown("## Histogramme")
    if st.checkbox("Voir les histogrammes"):
        for f in FEATURES:
            name = px.histogram(
                dataset[f][1:], marginal="violin", title=f
            )
            st.write(name)


def scatter_plot(dataset):
    name = dataset.columns[5:]
    st.markdown("## Scatter plot")
    if st.checkbox("Voir les scatter plot"):
        feature1 = st.selectbox("Feature 1:", name)
        feature2 = st.selectbox("Feature 2:", name)
        fig = px.scatter(
            dataset[1:],
            x=feature1,
            y=feature2,
        )
        st.write(fig)


def pair_plot(dataset):
    st.markdown("## Pair plot")
    if st.checkbox("Voir les pair plots"):
        fig_pair = px.scatter_matrix(dataset, dimensions = FEATURES, color="Hogwarts House")
        fig_pair.update_traces(diagonal_visible=False, showupperhalf=False)
        st.write(fig_pair)


def vizualisation(dataset):
    histogram(dataset)
    scatter_plot(dataset)
    pair_plot(dataset)
