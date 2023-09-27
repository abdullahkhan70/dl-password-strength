import os
import sys
import pandas as pd
import numpy as np
import scipy
import torch
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


@dataclass
class DataTransformationConfig:
    tfidf_vectorizer_file_path: str = os.path.join(
        'artifacts', 'tfidf_vectorizer.pkl')
    svd_file_path: str = os.path.join(
        'artifacts', 'svd.pkl')


@dataclass
class DataTransformer:

    def __init__(self) -> None:
        self.data_transformation_config = DataTransformationConfig()

    def get_device(self):
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        return device

    def get_transformer(self):
        return SimpleImputer(strategy='constant', missing_values='')

    def data_transforming(self, raw_path):

        try:
            # Load the training dataset.
            train_data = pd.read_csv(raw_path)

            # Pick the desired columns to train a model.
            x = train_data[["password"]]
            y = train_data[["strength"]]

            X_train, X_test, y_train, y_test = train_test_split(
                x, y, test_size=0.3, random_state=np.random.seed(42))

            # Print out the train and test dataset shapes.
            print(f"X_train Shape is: {X_train.shape}")
            print(f"y_train Shape is: {y_train.shape}")

            # Print out the train and test types.
            print(f"X_train Type is: {type(X_train)}")
            print(f"y_train Type is: {type(y_train)}")

            # Fit the training and testing dataset.
            imputer = self.get_transformer()
            X_train = imputer.fit_transform(
                X_train.values.reshape(-1, 1)).ravel()
            X_test = imputer.transform(
                X_test.values.reshape(-1, 1)).ravel()

            # Convert training and testing into TF-IDR
            vectorizer = TfidfVectorizer(analyzer='char')
            tfidf_X_train = vectorizer.fit_transform(
                X_train)
            tfidf_X_test = vectorizer.transform(X_test)

            # Print the Shape of Tf-Idf train and test:
            print(f"TF-IDF X_train shape is: {tfidf_X_train.shape}")
            print(f"TF-IDF X_train shape is: {tfidf_X_test.shape}")

            # Reducing the Dimensionality.
            svd = TruncatedSVD(
                n_components=1, random_state=np.random.seed(42))
            X_train = svd.fit_transform(tfidf_X_train)
            X_test = svd.transform(tfidf_X_test)

            print(f"TF-IDR Vectorizer Type: {type(tfidf_X_train)}")

            # Convert the X_train and y_train into Pytorch Tensor.
            device = self.get_device()

            save_object(
                self.data_transformation_config.tfidf_vectorizer_file_path,
                vectorizer
            )
            save_object(
                self.data_transformation_config.svd_file_path,
                svd
            )

            X_train = torch.tensor(X_train).float().to(device)
            X_test = torch.tensor(X_test).float().to(device)

            y_train = torch.tensor(np.ravel(y_train.values)).long().to(device)
            y_test = torch.tensor(np.ravel(y_test.values)).long().to(device)

            # Print out the converted X_train and y_train data shape.
            print(f"X_train Tensor Shape is: {X_train.shape}")
            print(f"X_test Tensor Shape is: {X_test.shape}")
            print(f"y_train Tensor Shape is: {y_train.shape}")

            # Print out the Types of converted X_train and y_train.
            print(f"X_train Tensor Type is: {type(X_train)}")
            print(f"y_train Tensor Type is: {type(y_train)}")

            return (X_train, X_test, y_train, y_test)

        except Exception as error:
            raise CustomException(error, sys)
