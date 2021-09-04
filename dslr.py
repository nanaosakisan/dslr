import streamlit as st
import SessionState 

from parse import parse_train
from describe import describe
from vizualisation import vizualisation
from logreg_train import logreg_train
from logreg_predict import logreg_predict

st.title("DSLR")

st.sidebar.title("Navigation")
radio = st.sidebar.radio(label="", options=["Visualisation", "Entrainement", "Prediction"])

# Normally you'd do this:
#session_state = SessionState.get(a=0, b=0)
# ...but since we're not importing SessionState, we'll just do:
session_state = SessionState.get(dataset=[], theta_train=[], min_=0, max_=0)  # Pick some initial values.

if radio == "Visualisation":
    filename = st.file_uploader("Fichier d'entainement :", type="csv")
    st.markdown("## Vizualisation")
    if filename :
        res, error, dataset = parse_train(filename)
        session_state.dataset = dataset
        if res == 1:
            col1, col2 = st.beta_columns(2)   

            st.markdown("### Dataset")
            st.dataframe(dataset)
            st.markdown("### Describe")
            des = describe(dataset)
            st.dataframe(des)
            vizualisation(dataset)
        else:
          st.write(error)

elif radio == "Entrainement":
    st.markdown("## Entrainement")
    if len(session_state.dataset) != 0 :
        iter_input = st.slider("Itérations", 0, 10000, 1500)
        alpha = st.slider("Alpha", 0.01, 0.1, 0.1)
        session_state.theta_train, session_state.min_, session_state.max_ = logreg_train(session_state.dataset, iter_input, alpha)
        # st.write("Theta ravenclaw: " + str(session_state.theta_train[0][0]) + ", " + \
        #     str(session_state.theta_train[0][1]) +", " + str(session_state.theta_train[0][2]))
        # st.write("Theta slytherin: " + str(session_state.theta_train[1][0]) + ", " + \
        #     str(session_state.theta_train[1][1]) +", " + str(session_state.theta_train[1][2]))
        # st.write("Theta gryffindor: " + str(session_state.theta_train[2][0]) + ", " + \
        #     str(session_state.theta_train[2][1]) +", " + str(session_state.theta_train[2][2]))
        # st.write("Theta hufflepuff: " + str(session_state.theta_train[3][0]) + ", " + \
        #     str(session_state.theta_train[3][1]) +", " + str(session_state.theta_train[3][2]))
elif radio == "Prediction":
    st.markdown("## Prediction")
    file_predict = st.file_uploader("Fichier de prédiction :", type="csv")
    if file_predict:
        res, error, predict = parse_train(file_predict)
        if res == 1:
            st.markdown("### Dataset de prédiction")
            st.dataframe(predict)
            house_predict = logreg_predict(predict, session_state.theta_train, session_state.min_, session_state.max_)
            st.markdown("### Prediction")
            st.dataframe(house_predict)
        else:
            st.write(error)
