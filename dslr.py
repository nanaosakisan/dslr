import streamlit as st
import SessionState 

from parse import parse_train
from describe import describe
from vizualisation import vizualisation
from logreg_train import logreg_train
from logreg_predict import logreg_predict

st.title("DSLR")

st.sidebar.title("Navigation")
radio = st.sidebar.radio(label="", options=["Visualisation", "Entrainement", "Add them"])

# Normally you'd do this:
#session_state = SessionState.get(a=0, b=0)
# ...but since we're not importing SessionState, we'll just do:
session_state = SessionState.get(a=0, b=0, dataset=[])  # Pick some initial values.

if radio == "Visualisation":
    session_state.a = float(st.text_input(label="What is a?", value=session_state.a))
    st.write(f"You set a to {session_state.a}")
    filename = st.file_uploader("Fichier d'entainement :", type="csv")
    if filename :
        res, error, dataset = parse_train(filename)
        session_state.dataset = dataset
        if res == 1:
            col1, col2 = st.beta_columns(2)   

            st.markdown("## Dataset")
            st.dataframe(dataset)
            st.markdown("## Describe")
            des = describe(dataset)
            st.dataframe(des)
            vizualisation(dataset)
        else:
          st.write(error)

elif radio == "Entrainement":
    session_state.b = float(st.text_input(label="What is b?", value=session_state.b))
    st.write(f"You set b to {session_state.b}")
    st.markdown("## Entrainement")
    if len(session_state.dataset) != 0 :
        theta_train = []
        if st.checkbox("Afficher entrainement"):
            iter_input = st.slider("Itérations", 0, 10000)
            alpha = st.slider("Alpha", 0.01, 0.1, 0.1)
            theta_train, min_, max_ = logreg_train(dataset, iter_input, alpha)
            st.write("Theta ravenclaw: " + str(theta_train[0][0]) + ", " + \
                str(theta_train[0][1]) +", " + str(theta_train[0][2]))
            st.write("Theta slytherin: " + str(theta_train[1][0]) + ", " + \
                str(theta_train[1][1]) +", " + str(theta_train[1][2]))
            st.write("Theta gryffindor: " + str(theta_train[2][0]) + ", " + \
                str(theta_train[2][1]) +", " + str(theta_train[2][2]))
            st.write("Theta hufflepuff: " + str(theta_train[3][0]) + ", " + \
                str(theta_train[3][1]) +", " + str(theta_train[3][2]))
elif radio == "Add them":
    st.write(f"a={session_state.a} and b={session_state.b}")
    button = st.button("Add a and b")
    if button:
        st.write(f"a+b={session_state.a+session_state.b}")

# filename = st.file_uploader("Fichier d'entainement :", type="csv")

# if filename :
#     res, error, dataset = parse_train(filename)
#     theta_train = []
#     if res == 1:
#         col1, col2 = st.beta_columns(2)   

#         st.markdown("## Dataset")
#         st.dataframe(dataset)
#         st.markdown("## Describe")
#         des = describe(dataset)
#         st.dataframe(des)
#         vizualisation(dataset)

#         st.markdown("## Entrainement")
#         if st.checkbox("Afficher entrainement"):
#             iter_input = st.slider("Itérations", 0, 10000)
#             alpha = st.slider("Alpha", 0.01, 0.1, 0.1)
#             theta_train, min_, max_ = logreg_train(dataset, iter_input, alpha)
#             st.write("Theta ravenclaw: " + str(theta_train[0][0]) + ", " + \
#                 str(theta_train[0][1]) +", " + str(theta_train[0][2]))
#             st.write("Theta slytherin: " + str(theta_train[1][0]) + ", " + \
#                 str(theta_train[1][1]) +", " + str(theta_train[1][2]))
#             st.write("Theta gryffindor: " + str(theta_train[2][0]) + ", " + \
#                 str(theta_train[2][1]) +", " + str(theta_train[2][2]))
#             st.write("Theta hufflepuff: " + str(theta_train[3][0]) + ", " + \
#                 str(theta_train[3][1]) +", " + str(theta_train[3][2]))

#         st.markdown("## Prediction")
#         if st.checkbox("Afficher prédiction"):
#             file_predict = st.file_uploader("Fichier de prédiction :", type="csv")
#             if file_predict:
#                 res, error, predict = parse_train(file_predict)
#                 if res == 1:
#                     st.dataframe(predict)
#                     house_predict = logreg_predict(predict, theta_train, min_, max_)
#                     st.dataframe(house_predict)
#                 else:
#                     st.write(error)
#     else:
#         st.write(error)
        