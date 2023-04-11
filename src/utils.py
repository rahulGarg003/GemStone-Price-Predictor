import pandas as pd
import os
import sys
import pickle

from src.logger import logging
from src.exception import CustomException

def load_dataframe_from_csv(file_path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        raise CustomException(e, sys)

def save_dataframe_to_csv(df: pd.DataFrame, file_path: str, index: bool, header: bool) -> None:
    try:
        df.to_csv(file_path, index=index, header=header)
    except Exception as e:
        raise CustomException(e, sys)

def save_object_as_pkl(file_path: str, obj):
    try:
        with open(file_path, 'wb') as file_obj:
            pickle.dump(
                obj=obj,
                file=file_obj
            )
    except Exception as e:
        CustomException(e, sys)