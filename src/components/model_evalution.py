import os
import sys
import torch
import pandas as pd
from dataclasses import dataclass
from model_trainer import TrainModel
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


@dataclass
class ModelEvalutionConfig:
    model_path: str = os.path.join('artifacts', 'model.pt')
    tfidf_vectorizer_file_path: str = os.path.join(
        'artifacts', 'tfidf_vectorizer.pkl')
    svd_file_path: str = os.path.join(
        'artifacts', 'svd.pkl')


class ModelEvalution:

    def __init__(self) -> None:
        self.model_evalution_config = ModelEvalutionConfig()

    def predict_pipeline(self, features, X_train: torch.TensorType):
        try:
            model = TrainModel(in_features=X_train.shape[1]).cuda()
            model.load_state_dict(torch.load(
                self.model_evalution_config.model_path))
            model.eval()

            tfidf_vectorizer = load_object(
                self.model_evalution_config.tfidf_vectorizer_file_path)
            svd = load_object(
                self.model_evalution_config.svd_file_path)

            tfidf = tfidf_vectorizer.transform([features])
            data_scaled = svd.transform(tfidf)
            data_scaled = torch.tensor(data_scaled).float().cuda()

            print(data_scaled)

            with torch.no_grad():
                pred = model(data_scaled)

            print(f"Predicatation is: {pred}")

        except Exception as error:
            raise CustomException(error, sys)
