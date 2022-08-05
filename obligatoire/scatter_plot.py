import argparse
import pandas as pd
import plotly.express as px


def scatter_plot(dataset: pd.DataFrame) -> None:
    fig = px.scatter(
        dataset,
        x="Arithmancy",
        y="Care of Magical Creatures",
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
        description="Display scatter plot with the features of the dataset"
    )
    parser.add_argument("dataset", type=str, help="Name of the train dataset")
    args = parser.parse_args()
    data = read_files(args)
    if data.size == 0:
        return
    print(f"Dataset\n{data}")
    scatter_plot(data)


main()
# Arithmancy and Care of magical Creature
