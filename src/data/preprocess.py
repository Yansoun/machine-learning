import pandas as pd
from sklearn.model_selection import train_test_split


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the churn dataset.
    """
    
    # remove duplicates
    df = df.drop_duplicates()

    # convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # handle missing values
    df = df.dropna()

    # convert target variable
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical variables.
    """
    df = df.drop("customerID", axis=1)
    df_encoded = pd.get_dummies(df, drop_first=True)

    return df_encoded


def split_data(df: pd.DataFrame):

    target = "Churn"

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test