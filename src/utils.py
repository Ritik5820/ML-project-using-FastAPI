import os
import sys

import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from tqdm import tqdm

from src.exception import CustomException


def split_data(df):
    """Split the input DataFrame into training and test sets.

    Args:
        df (pd.DataFrame): The input DataFrame to be split.

    Returns:
        tuple: A tuple containing the training set and test set DataFrames.
    """
    try:
        train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
        return train_set, test_set
    except Exception as e:
        raise CustomException(e, sys)


def save_object(file_path, obj):
    """Save a Python object to a file using dill serialization.

    Args:
        file_path (str): The file path where the object will be saved.
        obj (object): The Python object to be saved.

    Raises:
        CustomException: If an error occurs during the saving process.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(xtrain, xtest, ytrain, ytest, models, params):
    """Evaluate the performance of machine learning models using cross-validation.

    Args:
        xtrain (array-like): Training input data.
        xtest (array-like): Testing input data.
        ytrain (array-like): Training target values.
        ytest (array-like): Testing target values.
        models (dict): Dictionary of machine learning models.
        params (dict): Dictionary of hyperparameters for each model.

    Raises:
        CustomException: If an error occurs during the evaluation process.

    Returns:
        dict: A dictionary containing the R-squared scores for each model on the testing data.
    """
    try:
        report = {}

        for model_name, model in tqdm(models.items(), desc="Model Evaluation"):
            para = params[model_name]

            gs = GridSearchCV(model, para, cv=3)
            gs.fit(xtrain, ytrain)

            model.set_params(**gs.best_params_)
            model.fit(xtrain, ytrain)

            ytrain_pred = model.predict(xtrain)
            ytest_pred = model.predict(xtest)

            train_model_score = r2_score(ytrain, ytrain_pred)
            test_model_score = r2_score(ytest, ytest_pred)

            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """Load a serialized object from a file.

    Args:
        file_path (str): The path to the file containing the serialized object.

    Raises:
        CustomException: If an error occurs during the loading process.

    Returns:
        Any: The deserialized object.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
