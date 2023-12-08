import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging
from src.utils import split_data


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "raw.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered in data ingestion method or component successfully")
        try:
            logging.info("Reading dataset initiated")
            df = pd.read_csv("notebook\data\stud.csv")
            logging.info("Reading dataset completed")

            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True
            )

            logging.info("Creating raw.csv")
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Creation of raw.csv completed")

            logging.info("Calling split_data function from utils to split the data")
            train_set, test_set = split_data(df)
            logging.info("Split data call completed")

            logging.info("Converting splitted train and test data into csv")
            train_set.to_csv(
                self.ingestion_config.train_data_path, index=False, header=True
            )
            test_set.to_csv(
                self.ingestion_config.test_data_path, index=False, header=True
            )
            logging.info("Creation of train.csv and test.csv completed")

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e, sys)


# ----------------------------------For testing purpose only--------------------------------------------------:

# if __name__ == "__main__":
#     obj = DataIngestion()
#     train_data, test_data = obj.initiate_data_ingestion()

#     data_transformation = DataTransformation()

#     train_array, test_array, _ = data_transformation.initiate_data_transformation(
#         train_data, test_data
#     )

#     modeltraner = ModelTrainer()
#     print(
#         modeltraner.initiate_model_trainer(
#             train_array=train_array, test_array=test_array
#         )
#     )
