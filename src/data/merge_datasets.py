import pandas as pd
import os
RAW_DATA_DIR = "data/raw"
EXTERNAL_DATA_DIR = "data/external"

def load_fake_real_dataset():

    fake = pd.read_csv(os.path.join(EXTERNAL_DATA_DIR, "Fake.csv"))
    real = pd.read_csv(os.path.join(EXTERNAL_DATA_DIR, "True.csv"))

    fake["label"] = 0
    real["label"] = 1

    return fake, real


def load_gen_ai_dataset():

    path = os.path.join(EXTERNAL_DATA_DIR, "gen_ai_misinformation.csv")

    if os.path.exists(path):
        gen_ai = pd.read_csv(path)
        return gen_ai

    return None


def merge_datasets():

    fake, real = load_fake_real_dataset()

    df = pd.concat([fake, real], ignore_index=True)

    gen_ai = load_gen_ai_dataset()

    if gen_ai is not None:

        gen_ai = gen_ai.rename(columns={
            "text": "text",
            "label": "label"
        })

        df = pd.concat([df, gen_ai], ignore_index=True)

    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    df.to_csv(os.path.join(RAW_DATA_DIR, "news_dataset_raw.csv"), index=False)

    print("Dataset saved to data/raw/news_dataset_raw.csv")

    return df


if __name__ == "__main__":
    merge_datasets()