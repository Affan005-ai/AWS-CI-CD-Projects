import os 
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.logger import logging 
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.impute import SimpleImputer
from src.utils import save_object

@dataclass
class data_transformation_config:
    preprocessor_obj_file_path= os.path.join("artifacts", "preprocessor_1.pkl")


class Data_Transformation:
    def __init__(self):
        self.data_transformation_config= data_transformation_config()


    def get_data_transformer_object(self): 

        try:
            logging.info("Data Transformation initiated")

            numerical_columns= [ 'reading score', 'writing score']

            ordinal_columns = [
                "parental level of education",
                "lunch",
                "test preparation course"
            ]

            nominal_columns = ["gender", "race/ethnicity"]

            ordinal_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OrdinalEncoder(categories=[
                    ['some high school', 'high school', 'some college', "associate's degree", "bachelor's degree", "master's degree"],
                    ['free/reduced', 'standard'],
                    ['none', 'completed']
                ]))
            ])

            nominal_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ])

            num_pipeline= Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])


            logging.info(f"categorical columns :{ordinal_columns + nominal_columns}")
            logging.info(f"numerical columns :{numerical_columns}")

            preprocessor= ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("ordinal_pipeline", ordinal_pipeline, ordinal_columns),
                    ("nominal_pipeline", nominal_pipeline, nominal_columns) 
                ]
            )

            logging.info("Preprocessor pipeline created successfully")

            return preprocessor
        

        except Exception as e:
            logging.error("Error occurred in data transformation")
            raise CustomException(e, sys)
        

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df= pd.read_csv(train_path)
            test_df= pd.read_csv(test_path)
            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessor object")

            preprocessing_obj= self.get_data_transformer_object()

            target_column_name= "math score"
            numerical_columns= [ 'reading score', 'writing score']

            input_feature_train_df= train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df= train_df[target_column_name]

            input_feature_test_df= test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df= test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing dataframes")

            input_feature_train_arr= preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr= preprocessing_obj.transform(input_feature_test_df)

            train_arr= np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr= np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object")

            save_object(
                file_path= self.data_transformation_config.preprocessor_obj_file_path,
                obj= preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.error("Error occurred in data transformation")
            raise CustomException(e, sys)