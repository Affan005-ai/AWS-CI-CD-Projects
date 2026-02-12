print("🔥 File loaded")

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.exception import CustomException 
from dataclasses import dataclass
from src.components.data_transformation import Data_Transformation
from src.components.data_transformation import data_transformation_config

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer
from src.utils import save_object, load_object, evaluate_models



@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the Data Ingestion method")

        try:
            df = pd.read_csv("Notebook/data/StudentsPerformance.csv")
            logging.info("Dataset loaded")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Raw data saved")

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data ingestion completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.error("Error occurred in data ingestion")
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data ,test_data=obj.initiate_data_ingestion()

    data_transformation= Data_Transformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data, test_data)

    model_trainer= ModelTrainer()
    result= model_trainer.initiate_model_trainer(train_arr, test_arr)
    print(result)                    
            
                
        




