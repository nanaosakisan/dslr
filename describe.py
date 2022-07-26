from cmath import nan
import streamlit as st
import numpy as np
import math

FEATURES = ["Arithmancy"]
    # "Astronomy","Herbology","Defense Against the Dark Arts","Divination","Muggle Studies","Ancient Runes","History of Magic","Transfiguration","Potions","Care of Magical Creatures","Charms","Flying"]

def max_value(dataset, col):
    max_ = 0
    for row in dataset[1:]:
        if row[col] is not None and max_ < row[col]:
            max_ = row[col]
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


def repartition(dataset, count, col):
    sub_data = []
    for row in dataset[1:]:
        if row[col] is not None:
            sub_data.append(row[col])
    sub_data.sort()
    fist_quartile_ = fist_quartile(sub_data, count)
    mediane_ = mediane(sub_data, count)
    last_quartile_ = last_quartile(sub_data, count)
    return fist_quartile_, mediane_, last_quartile_


def min_value(dataset, col):
    min_ = 0
    for row in dataset[1:]:
        if row[col] is not None and min_ > row[col]:
            min_ = row[col]
    return min_


def std(data_list, mean, count, col):
    std = 0
    for row in data_list:
        if row is not None:
            std += (row - mean) ** 2
    return math.sqrt(std / count)


def mean(data_list, count):
    mean = 0
    for row in data_list:
        if row is not nan:
            mean += row
            print(row)
    print("mean")
    print(mean)
    return mean / count


def count(data_list):
    cpt = 0
    for row in data_list:
        if row is not None:
            cpt += 1
    return cpt


def push_feature(dataset, des, matiere):
    st.write(dataset)
    data_list = dataset.to_list()
    st.write(data_list)
    push = []
    push.append(matiere)
    count_ = count(data_list)
    push.append(count_)
    mean_ = mean(data_list, count_)
    print("mean_")
    print(mean_)
    push.append(mean_)
    # std_ = std(dataset, mean_, count_, col)
    # push.append(std_)
    # min_ = min_value(dataset, col)
    # push.append(min_)
    # fist_quartile, mediane_, last_quartile = repartition(dataset, count_, col)
    # push.append(fist_quartile)
    # push.append(mediane_)
    # push.append(last_quartile)
    # max_ = max_value(dataset, col)
    # push.append(max_)
    des.append(push)
    return des


def describe(dataset):
    des = [["Feature", "Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]]
    for f in FEATURES:
        des = push_feature(dataset[f], des, f)
    # des = push_feature(dataset, des, "Astronomy", 7)
    # des = push_feature(dataset, des, "Herbology", 8)
    # des = push_feature(dataset, des, "Defense Against the Dark Arts", 9)
    # des = push_feature(dataset, des, "Divination", 10)
    # des = push_feature(dataset, des, "Muggle Studies", 11)
    # des = push_feature(dataset, des, "Ancient Runes", 12)
    # des = push_feature(dataset, des, "History of Magic", 13)
    # des = push_feature(dataset, des, "Transfiguration", 14)
    # des = push_feature(dataset, des, "Potions", 15)
    # des = push_feature(dataset, des, "Care of Magical Creatures", 16)
    # des = push_feature(dataset, des, "Charms", 17)
    # des = push_feature(dataset, des, "Flying", 18)
    return des
