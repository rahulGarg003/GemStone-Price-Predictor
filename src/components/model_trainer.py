import os
import sys
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.config import Config

from src.utils import evaluate_model, save_object_as_pkl

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join(Config.z_artifacts_path, 'model.pkl')

class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info('Splitting dependent and independent variables in train and test')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            logging.info('Model Training Started')
            models = {
                'LinearRegression' : LinearRegression(), 
                'Lasso' : Lasso(), 
                'Ridge' : Ridge(), 
                'ElasticNet' : ElasticNet()
            }

            model_report: dict = evaluate_model(
                X_train=X_train, 
                y_train=y_train, 
                X_test=X_test, 
                y_test=y_test, 
                models=models
            )
            print('Model Report')
            print('\n', '='*50, '\n')
            logging.info(f'Model Report: {model_report}')

            R2_scores = [model['R2score_test'] for model in model_report.values()]
            best_model_score = max(R2_scores)
            best_model_name = list(model_report.keys())[R2_scores.index(best_model_score)]

            best_model = models[best_model_name]
            print(
                f'''
                    Best Model Found: 
                    \nModel Name: {best_model_name}
                    \n'mse_train' : {model_report[best_model_name]['mse_train']},
                    \n'mse_test' : {model_report[best_model_name]['mse_test']},
                    \n'absolute_mean_error_train' : {model_report[best_model_name]['absolute_mean_error_train']},
                    \n'absolute_mean_error_test' : {model_report[best_model_name]['absolute_mean_error_test']},
                    \n'rmse_train' : {model_report[best_model_name]['rmse_train']},
                    \n'rmse_test' : {model_report[best_model_name]['rmse_test']},
                    \n'R2score_train' : {model_report[best_model_name]['R2score_train']},
                    \n'R2score_test' : {model_report[best_model_name]['R2score_test']},
                    \n'adjusted_R2score_train' : {model_report[best_model_name]['adjusted_R2score_train']},
                    \n'adjusted_R2score_test' : {model_report[best_model_name]['adjusted_R2score_test']}
                '''
            )
            print('\n', '='*50)

            logging.info(
                f'''Best Model Found: 
                    \nModel Name: {best_model_name}
                    \n'mse_train' : {model_report[best_model_name]['mse_train']},
                    \n'mse_test' : {model_report[best_model_name]['mse_test']},
                    \n'absolute_mean_error_train' : {model_report[best_model_name]['absolute_mean_error_train']},
                    \n'absolute_mean_error_test' : {model_report[best_model_name]['absolute_mean_error_test']},
                    \n'rmse_train' : {model_report[best_model_name]['rmse_train']},
                    \n'rmse_test' : {model_report[best_model_name]['rmse_test']},
                    \n'R2score_train' : {model_report[best_model_name]['R2score_train']},
                    \n'R2score_test' : {model_report[best_model_name]['R2score_test']},
                    \n'adjusted_R2score_train' : {model_report[best_model_name]['adjusted_R2score_train']},
                    \n'adjusted_R2score_test' : {model_report[best_model_name]['adjusted_R2score_test']}
                '''
            )

            save_object_as_pkl(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
        except Exception as e:
            raise CustomException(e, sys)