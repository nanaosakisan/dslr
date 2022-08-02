import streamlit as st
import pandas as pd
import math

import utils.settings as settings

FEATURES_NUM = [
    "Arithmancy",
    "Astronomy",
    "Herbology",
    "Defense Against the Dark Arts",
    "Divination",
    "Muggle Studies",
    "Ancient Runes",
    "History of Magic",
    "Transfiguration",
    "Potions",
    "Care of Magical Creatures",
    "Charms",
    "Flying",
]

FEATURES_CAT = ["Hogwarts House", "First Name", "Last Name", "Best Hand"]

##pour les features categoriel : most_present, list_unique


def max_value(data_list):
    max_ = 0
    for row in data_list:
        if max_ < row:
            max_ = row
    return max_


def last_quartile(sub_data, count):
    return sub_data[int(0.75 * count) + 1]


def fist_quartile(sub_data, count):
    return sub_data[int(0.25 * count) + 1]


def mediane(sub_data, count):
    mediane = 0
    if count % 2 != 0:
        mediane = sub_data[int(count / 2 - 1)]
    else:
        id_ = int(count / 2 - 1)
        mediane = (sub_data[id_] + sub_data[id_ + 1]) / 2
    return mediane


def repartition(data_list, count):
    sub_data = data_list.copy()
    sub_data.sort()
    fist_quartile_ = fist_quartile(sub_data, count)
    mediane_ = mediane(sub_data, count)
    last_quartile_ = last_quartile(sub_data, count)
    return fist_quartile_, mediane_, last_quartile_


def min_value(data_list):
    min_ = 0
    for row in data_list:
        if min_ > row:
            min_ = row
    return min_


def std(data_list, mean, count):
    std = 0
    for row in data_list:
        std += (row - mean) ** 2
    return math.sqrt(std / count)


def mean(data_list, count):
    mean = 0
    for row in data_list:
        mean += row
    return mean / count


def unique(data_list):
    unique_list = []
    count = 0
    for row in data_list:
        if row not in unique_list:
            unique_list.append(row)
            count += 1
    return count, unique_list


def count(data_list):
    return len(data_list)


# def most_present(data_list, uni_list):
#     uni_count = []
#     for c, row in enumerate(data_list):


def push_feature_numerique(dataset, des, matiere):
    dataset = dataset.dropna()
    data_list = dataset.to_list()
    push = []
    push.append(matiere)
    count_ = count(data_list)
    push.append(count_)
    uni, uni_list = unique(data_list)
    push.append(uni)
    mean_ = mean(data_list, count_)
    push.append(mean_)
    std_ = std(data_list, mean_, count_)
    push.append(std_)
    min_ = min_value(data_list)
    push.append(min_)
    fist_quartile, mediane_, last_quartile = repartition(data_list, count_)
    push.append(fist_quartile)
    push.append(mediane_)
    push.append(last_quartile)
    max_ = max_value(data_list)
    push.append(max_)
    push.append(None)
    push.append(None)
    des.append(push)
    return des


def push_feature_cat(dataset, des, feature):
    dataset = dataset.dropna()
    data_list = dataset.to_list()
    push = []
    push.append(feature)
    count_ = count(data_list)
    push.append(count_)
    uni, uni_list = unique(data_list)
    push.append(uni)
    push.append(None)
    push.append(None)
    push.append(None)
    push.append(None)
    push.append(None)
    push.append(None)
    push.append(None)
    # most_ =
    push.append(None)
    push.append(uni_list)
    des.append(push)
    return des


def describe(dataset):
    des = [
        [
            "Feature",
            "Count",
            "Unique",
            "Mean",
            "Std",
            "Min",
            "25%",
            "50%",
            "75%",
            "Max",
            "Most present",
            "List categorie",
        ]
    ]
    for f in FEATURES_NUM:
        des = push_feature_numerique(dataset[f], des, f)
    for f in FEATURES_CAT:
        des = push_feature_cat(dataset[f], des, f)
    df = pd.DataFrame(des[1:], columns=des[0]).T
    return df.rename(columns=df.iloc[0]).drop(df.index[0])


st.title("Describe")
data = settings.dataset
if data.size == 0:
    st.error("Please upload a file.")
else:
    des = describe(data)
    st.dataframe(data)
    st.dataframe(des.astype(str))
