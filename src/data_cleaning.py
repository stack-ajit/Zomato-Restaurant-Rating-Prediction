
import pandas as pd


def load_raw_data(path: str) -> pd.DataFrame:
    
    df = pd.read_csv(path)
    print(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def parse_rate(value):
    """Convert a rate string like '4.1/5' or '4.1' into a float. Returns None if unparseable."""
    try:
        return float(value)
    except (TypeError, ValueError):
        try:
            if isinstance(value, str) and "/" in value:
                return float(value.split("/")[0])
        except (TypeError, ValueError):
            pass
    return None


def parse_cost(value):
    """Convert a cost string like '1,200' into a float. Returns None if unparseable."""
    try:
        cleaned = str(value).replace(",", "")
        return float(cleaned)
    except (TypeError, ValueError):
        return None


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline:
    - Drop rows missing key fields (location, cuisines, rest_type, cost)
    - Fill dish_liked nulls with a flag rather than dropping (too much missingness to drop)
    - Parse rate and cost into numeric columns
    - Drop columns not useful for analysis (url, address, menu_item, phone, listed_in(city))
    """
    df = df.copy()

   
    df = df.dropna(subset=["location", "cuisines", "rest_type", "approx_cost(for two people)"])

    # dish_liked is >50% missing -- too sparse to use as a real feature.
    # We keep a presence flag instead of dropping the column outright.
    df["has_dish_liked_info"] = df["dish_liked"].notna().astype(int)
    df = df.drop(columns=["dish_liked"])

    # Parse messy numeric-as-string columns
    df["rate"] = df["rate"].apply(parse_rate)
    df["approx_cost(for two people)"] = df["approx_cost(for two people)"].apply(parse_cost)

    # Drop columns that aren't useful for EDA or modeling
    drop_cols = ["url", "address", "menu_item", "listed_in(city)", "phone", "reviews_list"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Final: drop any rows where rate is still null (can't use as target if predicting rate)
    before = df.shape[0]
    df = df.dropna(subset=["rate"])
    print(f"Dropped {before - df.shape[0]} rows with missing rate (target column)")

    print(f"Cleaned data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def save_processed(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)
    print(f"Saved cleaned data to {path}")


if __name__ == "__main__":
    raw = load_raw_data("data/raw/zomato.csv")
    cleaned = clean_data(raw)
    save_processed(cleaned, "data/processed/zomato_clean.csv")
