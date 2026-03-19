import pandas as pd
from pathlib import Path


def load_raw_data(data_path: str) -> pd.DataFrame:
    """
    Load the raw churn dataset.

    Parameters
    ----------
    data_path : str
        Path to the raw dataset CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded dataframe.
    """

    df = pd.read_csv(data_path)

    print(f"Dataset loaded successfully with shape: {df.shape}")

    return df


def save_processed_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Save dataframe to processed data folder.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to save
    output_path : str
        Destination path
    """

    # create folder if it does not exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    print(f"Processed data saved to: {output_path}")