import os
import sys
import torch
import pandas as pd
from dataclasses import dataclass
from src.components.model_trainer import TrainModel
from src.exception import CustomException
from src.utils import load_object


@dataclass
class PipelinePredictConfig:
    model_path: str = os.path.join('artifacts', 'model.pt')
    tfidf_vectorizer_file_path: str = os.path.join(
        'artifacts', 'tfidf_vectorizer.pkl')
    svd_file_path: str = os.path.join(
        'artifacts', 'svd.pkl')
    train_preprocessor: str = os.path.join(
        'artifacts', 'train_preprocessor.pkl')


class PipelinePrediction:

    def __init__(self) -> None:
        self.pipeline_prediction_config = PipelinePredictConfig()

    def predict_pipeline(self, features):
        try:
            # Load the stored Preprocessors.
            X_train = load_object(
                self.pipeline_prediction_config.train_preprocessor)
            tfidf_vectorizer = load_object(
                self.pipeline_prediction_config.tfidf_vectorizer_file_path)
            svd = load_object(
                self.pipeline_prediction_config.svd_file_path)
            
            # Load the Existing Model.
            model = TrainModel(in_features=X_train.shape[1]).cuda()
            model.load_state_dict(torch.load(
                self.pipeline_prediction_config.model_path))
            model.eval()

            tfidf = tfidf_vectorizer.transform([str(features["password"]).lower()])
            data_scaled = svd.transform(tfidf)
            data_scaled = torch.tensor(data_scaled, requires_grad=True).float().cuda()

            print(data_scaled)

            pred = model(data_scaled).cuda().squeeze()[0]

            if pred >= 1:
                return "Strong Password"
            else:
                return "Weak Password"
            
        except Exception as error:
            raise CustomException(error, sys)

class CustomDataClass:
    def __init__(self, password: str) -> None:
        self.password = password
        
    def get_custom_data(self):
        try:
            custom_data_input_dict = {
                'password': [self.password]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as error:
            raise CustomDataClass(error, sys)