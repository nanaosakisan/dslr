import streamlit as st
import math

from preprocess_predict import preprocess_predict

def score(theta1, theta2, row):
    return theta1 * row[0] + theta2 * row[1]

def predict(row, theta_train):
    pred0 = score(theta_train[0][0], theta_train[0][1], row)
    pred1 = score(theta_train[1][0], theta_train[1][1], row)
    pred2 = score(theta_train[2][0], theta_train[2][1], row)
    pred3 = score(theta_train[3][0], theta_train[3][1], row)
    return [pred0, pred1, pred2, pred3]

def sigmoid(pred):
    proba0 = 1 / (1 + math.exp(-pred[0]))
    proba1 = 1 / (1 + math.exp(-pred[1]))
    proba2 = 1 / (1 + math.exp(-pred[2]))
    proba3 = 1 / (1 + math.exp(-pred[3]))
    return [proba0, proba1, proba2, proba3] 

def logreg_predict(dataset, theta_train, min_, max_):
    sub_data = preprocess_predict(dataset, 2 + 5, 3 + 5, min_, max_)
    st.markdown("### Dataset de prediction après le preprocess")
    st.dataframe(sub_data)

    house = {0:"Ravenclaw", 1:"Slytherin", 2:"Gryffindor", 3:"Hufflepuff"}

    house_predict = []

    st.write("Theta")
    st.dataframe(theta_train)
    # st.write("Theta0: ", theta_train[0][0])
    # st.write("Theta1: ", theta_train[0][1])
    # st.write("Theta2: ", theta_train[0][2])
    
    for i, row in enumerate(sub_data[1:100]):
        proba = sigmoid(predict(row, theta_train))
        st.dataframe(proba)
        index = proba.index(max(proba))
        push = [i, house.get(index)]
        # push = [i, index]
        house_predict.append(push)
    return house_predict
