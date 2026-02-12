import pandas as pd
import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
from pathlib import Path

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
         
            # ...
            BASE_DIR = Path(__file__).resolve().parents[2]  # project root
            ARTIFACTS_DIR = BASE_DIR / "artifacts"

            model_path = ARTIFACTS_DIR / "model_1.pkl"
            preprocessor_path = ARTIFACTS_DIR / "preprocessor_1.pkl"

            model = load_object(file_path=str(model_path))
            preprocessor = load_object(file_path=str(preprocessor_path))


            data_scaled=preprocessor.transform(features)

            preds=model.predict(data_scaled)
            return preds

        except Exception as e:
            logging.info("Exception occurred in prediction")
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
                 gender: str,
                 race_ethnicity: str,
                 parental_level_of_education: str,
                 lunch: str,
                 test_preparation_course: str,
                 reading_score: int,
                 writing_score: int):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race/ethnicity": [self.race_ethnicity],
                "parental level of education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test preparation course": [self.test_preparation_course],
                "reading score": [self.reading_score],
                "writing score": [self.writing_score]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            logging.info("Exception occurred in prediction pipeline")
            raise CustomException(e, sys)