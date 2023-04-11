import os
import sys
from dataclasses import dataclass

import numpy as np

from src.logger import logging
from src.exception import CustomException
from src.config import Config

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.utils import load_dataframe_from_csv, save_object_as_pkl

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join(Config.z_artifacts_path, 'preproccessor.pkl')

class DataTransfomation:
    def __init__(self) -> None:
        self.transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation started')
            
            # defining categorical cols and numerical cols
            categorical_features = ['cut', 'color', 'clarity']
            numerical_features = ['carat', 'depth', 'table', 'x', 'y', 'z']

            #defining custom Ranking to each oridinal variable 
            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['J', 'I', 'H', 'G', 'F', 'E', 'D']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            #defining pipeline for both type of varibales
            logging.info('Pipeline Initiated')

            num_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('encoder', OrdinalEncoder(categories=[cut_categories, color_categories, clarity_categories])),
                    ('scaler', StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, numerical_features),
                    ('cat_pipeline', cat_pipeline, categorical_features)
                ]
            )
            logging.info('Pipeline Completed')
            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_data_path: str, test_data_path: str):
        try:
            # loading train and test data
            train_df = load_dataframe_from_csv(train_data_path)
            test_df = load_dataframe_from_csv(test_data_path)

            logging.info('Reading of train and test data completed')
            logging.info(f'Train data: \n{train_df.head().to_string()}')
            logging.info(f'Test data: \n{test_df.head().to_string()}')

            logging.info('Obtaining Preprocessing Object')
            preprocessor_obj = self.get_data_transformation_object()

            target_column = 'price'
            drop_columns = [target_column, 'id']

            input_features_train_df = train_df.drop(labels=drop_columns, axis=1)
            target_feature_train_df = train_df[target_column]

            input_features_test_df = test_df.drop(labels=drop_columns, axis=1)
            target_feature_test_df = test_df[target_column]

            logging.info('Applying preprocessing steps on train and test independent features')

            ## transforming using preprocessor obj
            input_features_train_arr = preprocessor_obj.fit_transform(input_features_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_features_test_df)

            #converting df to numpy array
            train_arr = np.c_[input_features_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object_as_pkl(
                file_path=self.transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
            logging.info('Preproccessor object saved as pkl file')

            return(
                train_arr,
                test_arr,
                self.transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)

