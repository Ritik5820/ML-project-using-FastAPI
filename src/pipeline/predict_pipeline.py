import os
import sys

import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def predict(self, features):
        """Predict the target variable using a pre-trained model.

        Args:
            features (pd.DataFrame): Input features for prediction.

        Raises:
            CustomException: An exception raised during the prediction process.

        Returns:
            np.ndarray: Predicted target variable.
        """
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            model = load_object(file_path=model_path)
            data_ingestion_instance = DataIngestion()
            train_path, _ = data_ingestion_instance.initiate_data_ingestion()
            train_df = pd.read_csv(train_path)
            data_trans_instance = DataTransformation()
            preprocessor = data_trans_instance.get_data_transformer_object()
            train_arr = preprocessor.fit_transform(train_df)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)

            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int,
    ):
        """Initialize an instance of the PredictData class.

        Args:
            gender (str): Gender of the individual.
            race_ethnicity (str): Race or ethnicity of the individual.
            parental_level_of_education (str): Level of parental education.
            lunch (str): Type of lunch.
            test_preparation_course (str): Test preparation course information.
            reading_score (int): Reading score.
            writing_score (int): Writing score.
        """
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_dataframe(self):
        """Return data as a Pandas DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing the data attributes.

        Raises:
            CustomException: If an error occurs while creating the DataFrame.
        """
        try:
            custom_data_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_dict)

        except Exception as e:
            raise CustomException(e, sys)
