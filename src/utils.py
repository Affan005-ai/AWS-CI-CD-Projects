import os 
import sys
import pandas as pd
import numpy as np
import dill
import pickle
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV

def save_object(file_path, obj):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    


def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_models(X_train, y_train, X_test, y_test, models, params):

    try:
        report = {}
        trained_models = {}

        for model_name, model in models.items():
            logging.info(f"Training started for {model_name}")

            param_grid = params.get(model_name, {})

            # If hyperparameters exist → tune
            if param_grid:
                search = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=param_grid,
                    n_iter=9,
                    cv=3,
                    scoring="r2",
                    n_jobs=-1,
                    verbose=1
                )
                search.fit(X_train, y_train)

                best_model = search.best_estimator_
                logging.info(f"{model_name} best params: {search.best_params_}")

            else:
                model.fit(X_train, y_train)
                best_model = model

            # --- Test prediction ---
            y_test_pred = best_model.predict(X_test)
            test_score = r2_score(y_test, y_test_pred)

            # Store results
            report[model_name] = test_score
            trained_models[model_name] = best_model

            logging.info(f"{model_name} Test R2 Score: {test_score}")

        return report, trained_models

    except Exception as e:
        raise CustomException(e, sys)

