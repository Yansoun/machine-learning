import pandas as pd

from src.data.ingest import load_raw_data, save_processed_data
from src.data.preprocess import clean_data, encode_features, split_data


def run_data_pipeline():

    # Load raw data
    df = load_raw_data("data/raw/telco_churn.csv")

    # Clean dataset
    df = clean_data(df)

    # Encode categorical variables
    df = encode_features(df)

    # Split dataset
    X_train, X_test, y_train, y_test = split_data(df)

    # Save processed datasets
    save_processed_data(X_train, "data/processed/X_train.csv")
    save_processed_data(X_test, "data/processed/X_test.csv")
    save_processed_data(y_train, "data/processed/y_train.csv")
    save_processed_data(y_test, "data/processed/y_test.csv")


if __name__ == "__main__":
    run_data_pipeline()