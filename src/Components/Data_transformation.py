import sys
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from dataclasses import dataclass

from src.exception import customException
from src.logger import logging
from src.utils import save_obj

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()
    
    def get_data_transformer_obj(self):
        '''
        This function is responsible for returning the data transformer object.
        '''
        try:
            logging.info("Entered the data transformation module")
            numeric_features = ['Age', 'Number_of_Dependents', 'Work_Experience', 'Household_Size']
            categorical_features = ['Education_Level', 'Occupation', 'Location', 'Marital_Status', 'Employment_Status', 'Homeownership_Status', 'Type_of_Housing', 'Gender', 'Primary_Mode_of_Transportation']

            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical columns: {categorical_features}")
            logging.info(f"Numerical columns: {numeric_features}")

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numeric_features),
                ("cat_pipeline", cat_pipeline, categorical_features)
            ])

            return preprocessor
        except Exception as e:
            raise customException(e, sys)
       
    def initiate_data_transformation(self, train_path, test_path):
        '''
        This function is responsible for initiating the data transformation process.
        '''

        logging.info("Data transformation has started")

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Reading the train and test data completed")
            logging.info("Obtaining preprocessor object")

            preprocessing_obj =  self.get_data_transformer_obj()

            target_column_name = 'Income'
            numeric_columns = ['Age', 'Number_of_Dependents', 'Work_Experience', 'Household_Size']

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info("Applying the preprocesssor object on the train and test dataframe")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_obj(
                file_path = self.transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise customException(e,sys)