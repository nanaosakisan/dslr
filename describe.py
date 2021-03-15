import streamlit as st
import math

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
        mediane = sub_data[int(count/2 - 1)]
    else:
        id_ = int(count/2 - 1)
        mediane = (sub_data[id_] + sub_data[id_ + 1]) / 2
    return mediane

def repartition(dataset, count, col):
    sub_data = []
    for row in dataset[1:]:
        if row[col] is not None :
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

def std(dataset, mean, count, col) :
    std = 0
    for row in dataset[1:]:
        if row[col] is not None:
            std += (row[col] - mean)**2
    return math.sqrt(std / count)

def mean(dataset, count, col):
    mean = 0
    for row in dataset[1:]:
        if row[col] is not None:
            mean += row[col]
    return mean / count

def count(dataset, col) :
    cpt = 0
    for row in dataset[1:] :
        if row[col] is not None:
            cpt += 1
    return cpt

def push_feature(dataset, des, matiere, col) :
    push = []
    push.append(matiere)
    count_ = count(dataset, col)
    push.append(count_)
    mean_ = mean(dataset, count_, col)
    push.append(mean_)
    std_ = std(dataset, mean_, count_, col)
    push.append(std_)
    min_ = min_value(dataset, col)
    push.append(min_)
    fist_quartile, mediane_ , last_quartile = repartition(dataset, count_, col)
    push.append(fist_quartile)
    push.append(mediane_)
    push.append(last_quartile)
    max_ = max_value(dataset, col)
    push.append(max_)
    des.append(push)
    return des

def describe(dataset) :
    des = [["Feature", "Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]]
    des = push_feature(dataset, des,"Arithmancy", 6)
    des = push_feature(dataset, des,"Astronomy", 7)
    des = push_feature(dataset, des,"Herbology", 8)
    des = push_feature(dataset, des,"Defense Against the Dark Arts", 9)
    des = push_feature(dataset, des,"Divination", 10)
    des = push_feature(dataset, des,"Muggle Studies", 11)
    des = push_feature(dataset, des,"Ancient Runes", 12)
    des = push_feature(dataset, des,"History of Magic", 13)
    des = push_feature(dataset, des,"Transfiguration", 14)
    des = push_feature(dataset, des,"Potions", 15)
    des = push_feature(dataset, des,"Care of Magical Creatures", 16)
    des = push_feature(dataset, des,"Charms", 17)
    des = push_feature(dataset, des,"Flying", 18)
    return des