import pandas as pd
import os
import sys

def load_dataframe_from_csv(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)

def save_dataframe_to_csv(df: pd.DataFrame, file_path: str, index: bool, header: bool) -> None:
    df.to_csv(file_path, index=index, header=header)