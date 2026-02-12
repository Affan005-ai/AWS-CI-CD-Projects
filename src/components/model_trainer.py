import os 
import sys
import pandas as pd
from dataclasses import dataclass
from xgboost import XGBRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor, 
    AdaBoostRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score



from src.logger import logging  
from src.exception import CustomException
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model_1.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config= ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):

        try:
            logging.info("Splitting training and test input data")
            X_train, y_train = train_array[:,:-1], train_array[:,-1]
            X_test, y_test = test_array[:,:-1], test_array[:,-1]

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbours Regressor": KNeighborsRegressor(),
                "XGB Regressor": XGBRegressor(),
                "Ada Boost Regressor": AdaBoostRegressor()
            }

            params = {
                "Random Forest": {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                },
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter': ['best', 'random'],
                },
                "Gradient Boosting": {
                    'learning_rate': [0.01, 0.1, 0.2],
                    'n_estimators': [100, 200],
                },
                "Linear Regression": {},
                "K-Neighbours Regressor": {
                    'n_neighbors': [3, 5, 7],
                    'weights': ['uniform', 'distance'],
                },
                "XGB Regressor": {
                    'learning_rate': [0.01, 0.1, 0.2],
                    'n_estimators': [100, 200],
                },
                "Ada Boost Regressor": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                }
            }

            model_report,trained_models = evaluate_models(X_train, y_train, X_test, y_test, models, params)

            best_model_name = max(model_report, key=model_report.get)
            best_model_score = max(model_report.values())
            best_model = trained_models[best_model_name]

            

            logging.info(f"Best model: {best_model_name}")
            logging.info(f"Test R2 Score: {best_model_score}")

            # ---- Train score (to detect overfitting) ----
            y_train_pred = best_model.predict(X_train)
            train_score = r2_score(y_train, y_train_pred)

            logging.info(f"Train R2 Score: {train_score}")

            # Threshold check
            if best_model_score < 0.6:
                raise CustomException("No good model found", sys)

            

            


            logging.info(f"Best model found: {best_model_name} with R2 score: {best_model_score}")

            if best_model_score < 0.6:
                raise CustomException("No best model found with R2 score above threshold", sys)
            

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return {
                "model_name": best_model_name,
                "train_score": train_score,
                "test_score": best_model_score,
                "model path": self.model_trainer_config.trained_model_file_path
            }

        except Exception as e:
            raise CustomException(e, sys)