import os
import sys
import pandas as pd
from typing import List
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from data_transformer import DataTransformer
from model_trainer import ModelTrainer
from model_evalution import ModelEvalution
from sklearn.model_selection import train_test_split


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')
    train_preprocessor: str = os.path.join(
        'artifacts', 'train_preprocessor.pkl')


class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self) -> List[str]:
        logging.info(f"Data Ingestion is started!")

        try:
            data = pd.read_csv('research/data/passwords.csv')
            data.dropna(inplace=True)
            print(f"Data Shape: {data.shape}")
            print(f"Data Duplicated is: {data.duplicated().sum()}")
            logging.info(f"Read the Dataset.")
            os.makedirs(os.path.dirname(
                self.ingestion_config.train_data_path), exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path,
                        index=False, header=True)
            logging.info(f"Train and Test dataset split Initialized")
            trainning_dataset, testing_dataset = train_test_split(
                data, test_size=0.3, random_state=42)
            trainning_dataset.to_csv(
                self.ingestion_config.train_data_path, index=False, header=True)
            testing_dataset.to_csv(
                self.ingestion_config.test_data_path, index=False, header=True)
            logging.info(f"Ingestion of the data is completed!")
            return (self.ingestion_config.train_data_path, self.ingestion_config.test_data_path, self.ingestion_config.raw_data_path)
        except Exception as error:
            raise CustomException(error, sys)


if __name__ == "__main__":
    data_ingestion_config = DataIngestionConfig()
    data_ingestion = DataIngestion()
    data_transforms = DataTransformer()
    model_train = ModelTrainer()
    model_evalution = ModelEvalution()
    train_path, test_path, raw_path = data_ingestion.initiate_data_ingestion()
    X_train, X_test, y_train, y_test = data_transforms.data_transforming(
        raw_path=raw_path)

    save_object(data_ingestion_config.train_preprocessor, X_train)

    model_train.train_model(X_train=X_train, X_test=X_test,
                            y_train=y_train, y_test=y_test)
    model_evalution.predict_pipeline('mypassword', X_train)
