import plotly.express as px

import streamlit as st

def sub_data(dataset, col_grade):
    sub_data = [["Note", "Maison"]]
    for row in dataset[1:]:
        push = [row[col_grade], row[1]]
        sub_data.append(push)
    return sub_data

def histogram(dataset):
    sub_dataset = []
    sub_dataset.append(["Arithmancy", sub_data(dataset, 6)])
    sub_dataset.append(["Astronomy", sub_data(dataset, 7)])
    sub_dataset.append(["Herbology", sub_data(dataset, 8)])
    sub_dataset.append(["Defense against the dark arts", sub_data(dataset, 9)])
    sub_dataset.append(["Divination", sub_data(dataset, 10)])
    sub_dataset.append(["Muggle studies", sub_data(dataset, 11)])
    sub_dataset.append(["Ancient runes", sub_data(dataset, 12)])
    sub_dataset.append(["History of magic", sub_data(dataset, 13)])
    sub_dataset.append(["Transfiguration", sub_data(dataset, 14)])
    sub_dataset.append(["Potions", sub_data(dataset, 15)])
    sub_dataset.append(["Care of magical creatures", sub_data(dataset, 16)])
    sub_dataset.append(["Charms", sub_data(dataset, 17)])
    sub_dataset.append(["Flying", sub_data(dataset, 18)])

    st.markdown("### Histogramme")
    if st.checkbox("Voir les histogrammes"):
        for i, data in enumerate(sub_dataset) :
            name = "fig_" + str(i)
            dataframe = data[1]
            name = px.histogram(dataframe[1:], color=1, marginal="violin", title=data[0])
            st.write(name)

def scatter_plot(sub_data):
    name = ["Arithmancy", "Astronomy", "Herbology", "Defense Against the Dark Arts", \
        "Divination", "Muggle Studies", "Ancient Runes", "History of Magic", \
        "Transfiguration", "Potions", "Care of Magical Creatures", "Charms", "Flying"]
    st.markdown("### Scatter plot")
    if st.checkbox("Voir les scatter plot"):
        feature1 = st.selectbox("Feature 1:", name)
        feature2 = st.selectbox("Feature 2:", name)
        index1 = name.index(feature1) + 1
        index2 = name.index(feature2) + 1 
        fig = px.scatter(sub_data[1:], x=index1, y=index2, color=0, \
            labels={str(index1):feature1, str(index2):feature2})
        st.write(fig)

def pair_plot(sub_data) :
    features = list(range(1,14))
    name = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], ["Arithmancy", "Astronomy", \
        "Herbology", "Defense Against the Dark Arts", "Divination", "Muggle Studies", \
            "Ancient Runes", "History of Magic", "Transfiguration", "Potions", \
            "Care of Magical Creatures", "Charms", "Flying"]]
    st.markdown("### Pair plot")
    if st.checkbox("Voir les pair plots"):
        fig_pair = px.scatter_matrix(sub_data[1:], dimensions=features, color=0)
        fig_pair.update_traces(diagonal_visible=False, showupperhalf=False)
        st.dataframe(name)
        st.write(fig_pair)

def vizualisation(dataset):
    col = [1] + list(range(6, 19))
    sub_data = [[l[i] for i in col] for l in dataset]

    histogram(dataset)
    scatter_plot(sub_data)
    pair_plot(sub_data)
