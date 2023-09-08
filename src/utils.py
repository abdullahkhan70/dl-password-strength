import os
import sys
import dill
from .exception import CustomException
from .logger import logging


def save_object(file_path: str, obj: any = None):
    try:
        dirname = os.path.dirname(file_path)
        print(f"Saving the Model at: {dirname}")
        os.makedirs(dirname, exist_ok=True)

        with open(file_path, mode='wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as error:
        raise CustomException(error, sys)


def load_object(file_path: str):
    try:
        dir_name = os.path.dirname(file_path)
        print(f"File Path is: {dir_name}")

        with open(file_path, 'rb') as file_obj:
            data = dill.load(file=file_obj)

        return data

    except Exception as error:
        raise CustomException(error, sys)
