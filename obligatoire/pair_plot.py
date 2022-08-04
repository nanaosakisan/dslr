import argparse
import pandas as pd
from pathlib import Path
import plotly.express as px

FEATURES = [
    "Hogwarts House",
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


def check_features(name_col: list) -> bool:
    for f in FEATURES:
        if f not in name_col:
            return False
    return True


def pair_plot(dataset: pd.DataFrame) -> None:
    fig_pair = px.scatter_matrix(
        dataset,
        dimensions=FEATURES,
        color="Hogwarts House",
    )
    fig_pair.update_traces(diagonal_visible=False, showupperhalf=False)
    fig_pair.show()


def main():
    parser = argparse.ArgumentParser(
        description="Process some informations on the features of the dataset"
    )
    parser.add_argument("filename", type=str, help="Name of the train dataset")
    args = parser.parse_args()
    if Path(args.filename).suffix != ".csv":
        print("Please upload a csv file.")
        return
    data = pd.read_csv("./datasets/dataset_train.csv")
    if check_features(data.columns.to_list()) == False:
        print(f"Please upload a valid dataset with at least features : {FEATURES}")
    print(f"Dataset\n{data}")
    pair_plot(data)


main()
# Herbology and Defense Against the Dark Arts
