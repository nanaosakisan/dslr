import streamlit as st
import numpy as np
import pandas as pd
import time
import math
from numpy.random import rand

from preprocess_train import preprocess, convert_one_vs_all

# fonction cout J = -1/m * E((yp * log(ym)) + (1 - yp) * log(1 - yr))
# yr = y reel, yp = y predit

# gradient descent: theta1 -= a * (1 / m) * E((h(x) - y) *x1) 
# avec h(x) = 1 / (1 + exp(-A)) et A = theta0 + theta1 * x1 + theta2 * x2

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()        
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print ('%r  %2.2f ms' % \
                (method.__name__, (te - ts) * 1000))
        return result    
    return timed

class LogReg():
    def preprocess_(self, dataset):
        X_train = pd.DataFrame(dataset)
        X_train = X_train[[1, 7, 8]]
        X_train = X_train.dropna()
        # X_train = np.array(dataset)
        # X_train = np.delete(X_train, 0, 0)
        # X_train = X_train[:, [1, 7, 8]]
        # [print(row) for row in X_train]
        # print(X_train.dtype)
        # X_train = X_train[~np.isnan(X_train).any(axis=1)]
        # X_train = np.interp(X_train, (X_train.min(), X_train.max()), (-1, +1))
        y = []
        return X_train, y


    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def gradient(self, X_train, y, iter_input, lr):
        theta0_tmp, theta1_tmp, theta2_tmp = 0.0, 0.0, 0.0
        theta0, theta1, theta2 = 0.0, 0.0, 0.0
        thetas = []
        m = len(X_train)
        X_train = np.array(X_train)
        # for _i in range(0, iter_input) :
        #     sum0, sum1, sum2 = sum_(X_train, y, theta0, theta1, theta2)
        #     theta0_tmp -= lr * (1 / m) * sum0
        #     theta1_tmp -= lr * (1 / m) * sum1
        #     theta2_tmp -= lr * (1 / m) * sum2
        #     theta0, theta1, theta2 = theta0_tmp, theta1_tmp, theta2_tmp
        #     push = [theta0, theta1, theta2]
        #     thetas.append(push)
        # return theta0, theta1, theta2, thetas
        weight = rand(X_train.shape[1])
        for _ in range(0, 10):
            test = np.dot(X_train, weight)
            # y_pred = np.array([sigmoid(i) for i in test])
            y_pred = self.sigmoid(test)
            weight -= 0.05 * np.dot(X_train.T, y_pred-y)
        # st.write(weight)
        return weight

def score (theta0, theta1, theta2, row) :
    return theta0 + theta1 * row[0] + theta2 * row[1] 

def sum_(sub_data, y, theta0, theta1, theta2) :
    sum0, sum1, sum2 = 0, 0, 0
    for i, row in enumerate(sub_data):
        sig = sigmoid(score(theta0, theta1, theta2, row))
        sum0 += sig - y[i]
        sum1 += (sig - y[i]) * row[0]
        sum2 += (sig - y[i]) * row[1]
    return sum0, sum1, sum2

@timeit
def logreg_train(dataset, iter_input, alpha, **kwargs):
    st.write("Dataset train avant preprocess")
    st.dataframe(dataset)

    # sub_data, min_, max_ = preprocess(dataset, 2 + 5, 3 + 5)
    # st.write("Dataset train après preprocess")
    # st.dataframe(sub_data)


    # ravenclaw = convert_one_vs_all(sub_data, 0)
    # y_raven = [i.pop(0) for i in ravenclaw[1:]]
    # ravenclaw.pop(0)

    # slytherin = convert_one_vs_all(sub_data, 1)
    # y_slyth = [i.pop(0) for i in slytherin[1:]]
    # slytherin.pop(0)


    # gryffindor = convert_one_vs_all(sub_data, 2)
    # y_gryff = [i.pop(0) for i in gryffindor[1:]]
    # gryffindor.pop(0)

    # hufflepuff = convert_one_vs_all(sub_data, 3)
    # y_huffle = [i.pop(0) for i in hufflepuff[1:]]
    # hufflepuff.pop(0)


    # st.write("Dataset ravenclaw:")
    # st.dataframe(ravenclaw)
    # st.write("Dataset gynffindor:")
    # st.dataframe(gryffindor)

    log_train, y = LogReg().preprocess_(dataset)
    st.write("Dataset train après preprocess")
    st.dataframe(log_train)
    st.dataframe(y)

    # theta_train = []
    # push = gradient(ravenclaw, y_raven, iter_input, min_, max_, alpha)

    # theta_train.append(push)
    # push = gradient(slytherin, y_slyth, iter_input, min_, max_, alpha)

    # theta_train.append(push)
    # push = gradient(gryffindor, y_gryff, iter_input, min_, max_, alpha)

    # theta_train.append(push)
    # push = gradient(hufflepuff, y_huffle, iter_input, min_, max_, alpha)

    # theta_train.append(push)
    # st.write(theta_train)
    return theta_train, min_, max_
