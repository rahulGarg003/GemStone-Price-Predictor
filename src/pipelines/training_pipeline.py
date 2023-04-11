import os
import sys

from src.logger import logging
from src.exception import CustomException

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransfomation
from src.components.model_trainer import ModelTrainer

class TrainingPipeline:
    def __init__(self) -> None:
        pass

    def initiate_training_pipeline(self):
        ingestion = DataIngestion()
        train_Data_path, test_data_path = ingestion.initiate_data_ingestion()

        transformation = DataTransfomation()
        train_arr, test_arr, _ = transformation.initiate_data_transformation(
            train_data_path=train_Data_path, 
            test_data_path=test_data_path
        )

        trainer = ModelTrainer()
        trainer.initiate_model_training(
            train_array=train_arr,
            test_array=test_arr
        )

        