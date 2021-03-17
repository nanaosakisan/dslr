import streamlit as st
import time

from preprocess_train import preprocess

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

# def sum(sub_data, theta0, theta1) :
#     sum0, sum1 = 0, 0
#     for row in sub_data :
#         sum0 += ((theta0 + theta1 * row[1], theta2 * row[2]) - y) * row[1]
#         sum1 += ((theta0 + theta1 * row[1], theta2 * row[2]) - y) * row[2]
#     return sum0, sum1

# @timeit
# def gradient(sub_data, iter_input, min_data, max_data, alpha, **kwargs) :
#     theta0_tmp, theta1_tmp = 0, 0
#     thetas = []
#     m = len(sub_data)
#     for _i in range(0, iter_input) :
#         sum0, sum1 = sum(sub_data, theta0_tmp, theta1_tmp)
#         theta0_tmp -= alpha * (1 / m) * sum0
#         theta1_tmp -= alpha * (1 / m) * sum1
#         push = [theta0_tmp * (max_data - min_data) + min_data, theta1_tmp]
#         thetas.append(push)
#     return theta0_tmp, theta1_tmp, thetas

def convert_one_vs_all(dataframe, feature):
    st.write("You're in convert")

@timeit
def logreg_train(dataset, **kwargs):
    st.markdown("## Entrainement")
    sub_data = preprocess(dataset, 2 + 5, 3 + 5)
    st.dataframe(sub_data)
    ravenclaw = convert_one_vs_all(sub_data, 0)
    slytherin = convert_one_vs_all(sub_data, 1)
    gryffindor = convert_one_vs_all(sub_data, 2)
    hufflepuff = convert_one_vs_all(sub_data, 3)
