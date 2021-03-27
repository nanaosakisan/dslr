import streamlit as st
import time
import math

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

def score(row, theta0, theta1, theta2):
    return theta0 + theta1 * row[1] + theta2 * row[2]

def sum_(sub_data, theta0, theta1, theta2) :
    sum0, sum1, sum2 = 0, 0, 0
    for row in sub_data[1:] :
        sum0 += (1/(1 + math.exp(-score(row, theta0, theta1, theta2)))) - row[0]
        sum1 += ((1/(1 + math.exp(-(score(row, theta0, theta1, theta2))))) - row[0]) * row[1]
        sum2 += ((1/(1 + math.exp(-(score(row, theta0, theta1, theta2))))) - row[0]) * row[2]
    return sum0, sum1, sum2

def gradient(sub_data, iter_input, min_data, max_data, alpha) :
    theta0_tmp, theta1_tmp, theta2_tmp = 0.0, 0.0, 0.0
    theta0, theta1, theta2 = 0.0, 0.0, 0.0
    thetas = []
    m = len(sub_data)
    for _i in range(0, iter_input) :
        sum0, sum1, sum2 = sum_(sub_data, theta0, theta1, theta2)
        theta0_tmp -= alpha * (1 / m) * sum0
        theta1_tmp -= alpha * (1 / m) * sum1
        theta2_tmp -= alpha * (1 / m) * sum2
        theta0, theta1, theta2 = theta0_tmp, theta1_tmp, theta2_tmp
        # push = [theta0_tmp * (max_data - min_data) + min_data, theta1_tmp]
        # thetas.append(push)
    return theta0, theta1, theta2, thetas

@timeit
def logreg_train(dataset, iter_input, alpha, **kwargs):
    sub_data, min_, max_ = preprocess(dataset, 2 + 5, 3 + 5)
    st.write("Dataset train après preprocess")
    st.dataframe(sub_data)

    ravenclaw = convert_one_vs_all(sub_data, 0)
    slytherin = convert_one_vs_all(sub_data, 1)
    gryffindor = convert_one_vs_all(sub_data, 2)
    hufflepuff = convert_one_vs_all(sub_data, 3)

    st.write("Dataset ravenclaw:")
    st.dataframe(ravenclaw)
    st.write("Dataset slytherin:")
    st.dataframe(slytherin)

    
    theta_train = []
    push = gradient(ravenclaw, iter_input, min_, max_, alpha)
    theta_train.append(push)
    push = gradient(slytherin, iter_input, min_, max_, alpha)
    theta_train.append(push)
    push = gradient(gryffindor, iter_input, min_, max_, alpha)
    theta_train.append(push)
    push = gradient(hufflepuff, iter_input, min_, max_, alpha)
    theta_train.append(push)

    return theta_train, min_, max_
