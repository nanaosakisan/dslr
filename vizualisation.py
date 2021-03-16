import plotly.express as px

import streamlit as st

def sub_data(dataset, col_grade):
    sub_data = [["Note", "Maison"]]
    for row in dataset[1:]:
        push = [row[col_grade], row[1]]
        sub_data.append(push)
    return sub_data

def histogram(sub_dataset):
    st.markdown("## Histogramme")
    if st.checkbox("Voir les histogrammes"):
        for i, data in enumerate(sub_dataset) :
            name = "fig_" + str(i)
            dataframe = data[1]
            name = px.histogram(dataframe[1:], color=1, marginal="violin", title=data[0])
            st.write(name)

def scatter_plot(sub_dataset):
    st.markdown("## Scatter plot")
    if st.checkbox("Voir les scatter plot"):
        data=sub_dataset[1]
        dataframe = data[1][1:]
        st.dataframe(dataframe)
        # fig = px.scatter(x=dataframe)
        # st.write(fig)
        # for i, data in enumerate(sub_dataset) :
        #     name = "fig_" + str(i)
        #     dataframe = data[1]
        #     name = px.scatter(x = dataframe[0], y = dataframe[0], title=data[0])
        #     st.write(name)

def pair_plot(dataset) :
    st.markdown("## Pair plot")
    col = [1] + list(range(6, 19))
    features = list(range(1,14))
    name = ["Arithmancy", "Astronomy", "Herbology", "Defense Against the Dark Arts", \
        "Divination", "Muggle Studies", "Ancient Runes", "History of Magic", \
        "Transfiguration", "Potions", "Care of Magical Creatures", "Charms", "Flying"]
    sub_data = [[l[i] for i in col] for l in dataset]

    if st.checkbox("Voir les pair plots"):
        fig_pair = px.scatter_matrix(sub_data[1:], dimensions=features, color=0, \
            labels=name)
        fig_pair.update_traces(diagonal_visible=False, showupperhalf=False)
        st.dataframe(sub_data)
        st.write(fig_pair)


    # fig = px.scatter_matrix(df,
    # dimensions=["sepal_width", "sepal_length", "petal_width", "petal_length"],
    # color="species", symbol="species",
    # title="Scatter matrix of iris data set",
    # labels={col:col.replace('_', ' ') for col in df.columns}) # remove underscore
    # fig.update_traces(diagonal_visible=False)

 
    # data = sub_dataset[1][1]
    # st.dataframe(data)
    # fig = px.scatter_matrix(sub_dataset)
    # st.write(fig)

def vizualisation(dataset):
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

    histogram(sub_dataset)
    scatter_plot(sub_dataset)
    pair_plot(dataset)
