import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging


class DataTransformation:
    def get_data_transformer_object(self):
        """Get the data transformer object.

        Raises:
            CustomException: If an error occurs during the process.

        Returns:
            ColumnTransformer: Data transformer object.
        """
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(drop="first")),
                ]
            )
            logging.info("Numerical columns standard scaling completed")

            logging.info("Categorical columns encoding completed")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns),
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """Initiate data transformation.

        Args:
            train_path (str): Path to the training data.
            test_path (str): Path to the test data.

        Raises:
            CustomException: If an error occurs during the process.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Transformed training and test arrays.
        """
        try:
            logging.info("Reading train and test data initiated")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Reading train and test data completed")

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()
            logging.info("Preprocessor object obtained successfully")

            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                "Applying preprocessing object on training dataframe and testing dataframe"
            )
            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df
            )
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            logging.info(
                "Successfully applied preprocessing object on train and test dataframe"
            )

            logging.info("Concatenating features and labels for train and test dataset")
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            logging.info(
                "Concatenation features and labels for train and test dataset complete"
            )

            return (train_arr, test_arr)

        except Exception as e:
            raise CustomException(e, sys)
