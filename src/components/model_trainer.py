import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models, save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        """
        Initialize the ModelTrainer object.
        """
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        """Initialize the model training process.

        Args:
            train_array (numpy.ndarray): Training data array.
            test_array (numpy.ndarray): Test data array.

        Raises:
            CustomException: Raised for various exceptions.

        Returns:
            float: R2 score of the best model on the test dataset.
        """
        try:
            logging.info("Splitting into training and test input data")
            xtrain, ytrain, xtest, ytest = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            model = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "CatBoost": CatBoostRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "XGBoost": XGBRegressor(),
            }

            params = {
                "Decision Tree": {
                    "criterion": [
                        "squared_error",
                        "friedman_mse",
                        "absolute_error",
                        "poisson",
                    ],
                    "splitter": ["best", "random"],
                    "max_features": ["sqrt", "log2"],
                },
                "Random Forest": {
                    "criterion": [
                        "squared_error",
                        "friedman_mse",
                        "absolute_error",
                        "poisson",
                    ],
                    "max_features": ["sqrt", "log2", None],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "Gradient Boosting": {
                    "loss": ["squared_error", "huber", "absolute_error", "quantile"],
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    "criterion": ["squared_error", "friedman_mse"],
                    "max_features": ["auto", "sqrt", "log2"],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "Linear Regression": {},
                "XGBoost": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "CatBoost": {
                    "depth": [6, 8, 10],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "iterations": [30, 50, 100],
                },
                "AdaBoost": {
                    "learning_rate": [0.1, 0.01, 0.5, 0.001],
                    "loss": ["linear", "square", "exponential"],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
            }

            model_report: dict = evaluate_models(
                xtrain=xtrain,
                xtest=xtest,
                ytrain=ytrain,
                ytest=ytest,
                models=model,
                params=params,
            )
            logging.info("Finding the best model name with score initiated")

            best_model_score = max(sorted(model_report.values()))
            if best_model_score < 0.6:
                raise CustomException("Best model not found")

            for key in model_report.keys():
                if model_report[key] == max(sorted(model_report.values())):
                    best_model = model[key]

            logging.info("Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            predicted = best_model.predict(xtest)
            r2_sc = r2_score(ytest, predicted)

            return r2_sc

        except Exception as e:
            raise CustomException(e, sys)
