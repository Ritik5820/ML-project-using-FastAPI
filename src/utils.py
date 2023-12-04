import sys
from sklearn.model_selection import train_test_split
from src.exception import CustomException


def split_data(df):
    try:
        train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
        return train_set, test_set
    except Exception as e:
        raise CustomException(e, sys)
