import pandas as pd
import math
import argparse
from typing import Tuple, Optional


NUM_DES = [
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
]
CAT_DES = [
    "Feature",
    "Count",
    "Unique",
    "Most present",
    "Values",
]


def max_value(data_list: list) -> float:
    max_ = 0
    for row in data_list:
        if max_ < row:
            max_ = row
    return max_


def last_quartile(sub_data: list, count: int) -> float:
    return sub_data[int(0.75 * count) + 1]


def fist_quartile(sub_data: list, count: int) -> float:
    return sub_data[int(0.25 * count) + 1]


def mediane(sub_data: list, count: int) -> float:
    mediane = 0
    if count % 2 != 0:
        mediane = sub_data[int(count / 2 - 1)]
    else:
        id_ = int(count / 2 - 1)
        mediane = (sub_data[id_] + sub_data[id_ + 1]) / 2
    return mediane


def repartition(data_list: list, count: int) -> Optional[Tuple[float, float, float]]:
    if count == 0:
        return None, None, None
    sub_data = data_list.copy()
    sub_data.sort()
    fist_quartile_ = fist_quartile(sub_data, count)
    mediane_ = mediane(sub_data, count)
    last_quartile_ = last_quartile(sub_data, count)
    return fist_quartile_, mediane_, last_quartile_


def min_value(data_list: list) -> float:
    min_ = 0
    for row in data_list:
        if min_ > row:
            min_ = row
    return min_


def std(data_list: list, mean: float, count: int) -> Optional[float]:
    if count == 0:
        return None
    std = 0
    for row in data_list:
        std += (row - mean) ** 2
    return math.sqrt(std / count)


def mean(data_list: list, count: int) -> Optional[float]:
    if count == 0:
        return None
    mean = 0
    for row in data_list:
        mean += row
    return mean / count


def unique(data_list: list) -> Tuple[int, list]:
    unique_list = []
    count = 0
    for row in data_list:
        if row not in unique_list:
            unique_list.append(row)
            count += 1
    return count, unique_list


def count(data_list):
    return len(data_list)


def most_present(data_list: list, uni_list: list) -> str:
    uni_count = []
    for f in uni_list:
        count = data_list.count(f)
        uni_count.append([f, count])
    uni_count = sorted(uni_count)
    return uni_count[0][0]


def push_feature_numerique(dataset: pd.DataFrame, des: list, feature: str) -> list:
    dataset = dataset.dropna()
    data_list = dataset.to_list()
    push = []
    push.append(feature)
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
    des.append(push)
    return des


def push_feature_cat(dataset: pd.DataFrame, des: list, feature: str) -> list:
    dataset = dataset.dropna()
    data_list = dataset.to_list()
    push = []
    push.append(feature)
    count_ = count(data_list)
    push.append(count_)
    uni, uni_list = unique(data_list)
    push.append(uni)
    most_ = most_present(data_list, uni_list)
    push.append(most_)
    push.append(uni_list)
    des.append(push)
    return des


def describe(dataset: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    des_num = [NUM_DES]
    des_cat = [CAT_DES]

    types = dataset.dtypes
    cat_features = types.where(lambda x: x == object).dropna().index.tolist()
    num_features = types.where(lambda x: x != object).dropna().index.tolist()

    for f in num_features:
        des_num = push_feature_numerique(dataset[f], des_num, f)
    for f in cat_features:
        des_cat = push_feature_cat(dataset[f], des_cat, f)
    df_num = pd.DataFrame(des_num[1:], columns=des_num[0]).T
    df_cat = pd.DataFrame(des_cat[1:], columns=des_cat[0]).T
    return df_num.rename(columns=df_num.iloc[0]).drop(df_num.index[0]), df_cat.rename(
        columns=df_cat.iloc[0]
    ).drop(df_cat.index[0])


def read_files(args) -> pd.DataFrame:
    try:
        data = pd.read_csv(args.dataset, index_col=0)
    except Exception as e:
        print(f"{e.__class__} while reading dataset, please upload a valid file.")
        return pd.DataFrame()
    data_schema = pd.DataFrame(pd.io.json.build_table_schema(data).get("fields"))
    true_schema = pd.read_json("./obligatoire/utils/schema.json")
    if data_schema.equals(true_schema) != True:
        print("Schema error, please upload a valid file.")
        print(f"data_schema\n{data_schema}")
        print(f"true_schema\n{true_schema}")
        return pd.DataFrame()
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Process some informations on the features of the dataset"
    )
    parser.add_argument("dataset", type=str, help="Name of the train dataset")
    args = parser.parse_args()
    data = read_files(args)
    if data.size == 0:
        return
    print(f"Dataset\n{data}")
    des_num, des_cat = describe(data)
    print(f"Describe\nNumerical features\n{des_num}\nCategorical features\n{des_cat}")


main()
