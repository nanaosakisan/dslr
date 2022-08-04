import argparse
import pandas as pd
from pathlib import Path
import plotly.express as px

FEATURES = [
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


def scatter_plot(dataset: pd.DataFrame) -> None:
    fig = px.scatter(
        dataset,
        x="Arithmancy",
        y="Care of Magical Creatures",
    )
    fig.show()


def main():
    parser = argparse.ArgumentParser(
        description="Process some informations on the features of the dataset"
    )
    parser.add_argument("dataset", type=str, help="Name of the train dataset")
    args = parser.parse_args()
    if Path(args.dataset).suffix != ".csv":
        print("Please upload a csv file.")
        return
    data = pd.read_csv(args.dataset)
    if check_features(data.columns.to_list()) == False:
        print(f"Please upload a valid dataset with at least features : {FEATURES}")
    print(f"Dataset\n{data}")
    scatter_plot(data)


main()
# Arithmancy and Care of magical Creature
