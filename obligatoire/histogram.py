import argparse
import pandas as pd
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


def histogram(dataset: pd.DataFrame) -> None:
    for f in FEATURES:
        fig = px.histogram(
            dataset[["Hogwarts House", f]],
            x=f,
            marginal="violin",
            title=f,
            color="Hogwarts House",
        )
        fig.show()


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
        description="Display histogram with the features of the dataset"
    )
    parser.add_argument("dataset", type=str, help="Name of the train dataset")
    args = parser.parse_args()
    data = read_files(args)
    if data.size == 0:
        return
    print(f"Dataset\n{data}")
    histogram(data)


main()
## Arithmancy, Care of Magical Creatures
