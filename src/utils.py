import sys
import os
import dill

from tqdm import tqdm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
from src.exception import CustomException


def split_data(df):
    try:
        train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
        return train_set, test_set
    except Exception as e:
        raise CustomException(e, sys)


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(xtrain, xtest, ytrain, ytest, models, params):
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
