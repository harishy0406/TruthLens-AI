import pandas as pd
DATA_PATH = "data/raw/news_dataset_raw.csv"
def validate_dataset():

    df = pd.read_csv(DATA_PATH)

    print("\nDataset Shape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns)

    print("\nLabel Distribution:")
    print(df["label"].value_counts())

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDuplicate Rows:")
    print(df.duplicated().sum())


if __name__ == "__main__":
    validate_dataset()