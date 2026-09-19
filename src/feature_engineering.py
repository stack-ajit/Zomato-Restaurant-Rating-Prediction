

import numpy as np
import pandas as pd

TOP_N_CUISINES = 15


def binary_encode(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["online_order"] = (df["online_order"] == "Yes").astype(int)
    df["book_table"] = (df["book_table"] == "Yes").astype(int)
    return df


# def log_transform_votes(df: pd.DataFrame) -> pd.DataFrame:
#     df = df.copy()
#     df["votes_log"] = np.log1p(df["votes"])
#     return df


def frequency_encode(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Replace each category with how often it appears in the dataset."""
    df = df.copy()
    freq = df[column].value_counts(normalize=True)
    df[f"{column}_freq"] = df[column].map(freq)
    return df


def encode_cuisines(df: pd.DataFrame, top_n: int = TOP_N_CUISINES) -> pd.DataFrame:
    """
    Cuisines is a comma-separated multi-label field (e.g. 'North Indian, Chinese').
    We create:
      - cuisine_count: how many cuisines a restaurant serves
      - one binary column per top-N most common cuisine
    """
    df = df.copy()
    cuisine_lists = df["cuisines"].str.split(",").apply(lambda lst: [c.strip() for c in lst])
    df["cuisine_count"] = cuisine_lists.apply(len)

    all_cuisines = cuisine_lists.explode()
    top_cuisines = all_cuisines.value_counts().head(top_n).index.tolist()

    for cuisine in top_cuisines:
        col_name = f"cuisine_{cuisine.replace(' ', '_').lower()}"
        df[col_name] = cuisine_lists.apply(lambda lst: int(cuisine in lst))

    return df


def encode_listed_in_type(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dummies = pd.get_dummies(df["listed_in(type)"], prefix="type", drop_first=True).astype(int)
    df = pd.concat([df, dummies], axis=1)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature engineering pipeline and return a model-ready dataframe."""
    df = binary_encode(df)
  
    df = frequency_encode(df, "location")
    df = frequency_encode(df, "rest_type")
    df = encode_cuisines(df)
    df = encode_listed_in_type(df)

    feature_cols = (
        ["online_order", "book_table",  "approx_cost(for two people)",
         "location_freq", "rest_type_freq", "cuisine_count", "has_dish_liked_info"]
        + [c for c in df.columns if c.startswith("cuisine_") and c != "cuisine_count"]
        + [c for c in df.columns if c.startswith("type_")]
    )

    model_df = df[feature_cols + ["rate"]].copy()
    print(f"Feature matrix ready: {model_df.shape[0]} rows, {len(feature_cols)} features")
    return model_df


if __name__ == "__main__":
    df = pd.read_csv("data/processed/zomato_clean.csv")
    model_df = build_features(df)
    model_df.to_csv("data/processed/zomato_features.csv", index=False)
    print("Saved feature matrix to data/processed/zomato_features.csv")
