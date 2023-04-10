import os
import sys
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException

from src.utils import load_dataframe_from_csv, save_dataframe_to_csv

from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join(os.environ['Z_BASE_PATH'],'artifacts','raw.csv')
    train_data_path: str = os.path.join(os.environ['Z_BASE_PATH'],'artifacts','train.csv')
    test_data_path: str = os.path.join(os.environ['Z_BASE_PATH'],'artifacts','test.csv')


class DataIngestion:

    def __init__(self) -> None:
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion Started')
        try:
            data_url = r'https://raw.githubusercontent.com/krishnaik06/FSDSRegression/main/notebooks/data/gemstone.csv'
            df = load_dataframe_from_csv(data_url)
            logging.info('data read as pandas dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            save_dataframe_to_csv(
                df = df,
                file_path=self.ingestion_config.raw_data_path,
                index=False,
                header=True
            )

            logging.info('Train Test Split')
            train_set, test_set = train_test_split(df, test_size=0.3, random_state=42)
        
            save_dataframe_to_csv(
                df = train_set,
                file_path=self.ingestion_config.train_data_path,
                index=False,
                header=True
            )

            save_dataframe_to_csv(
                df = test_set,
                file_path=self.ingestion_config.test_data_path,
                index=False,
                header=True
            )
            logging.info('Data Ingestion Completed')
            
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)