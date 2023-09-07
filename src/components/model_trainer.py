import os
import sys
import pandas as pd
import torch
from torch import nn
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from dataclasses import dataclass


@dataclass
class ModelTrainerConfig:
    model_file_path: str = os.path.join('artifacts', 'model.pth')


class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerConfig()

    def train_model(self,
                    X_train: torch.TensorType,
                    X_test: torch.TensorType,
                    y_train: torch.TensorType,
                    y_test: torch.TensorType):
        try:
            df = pd.read_csv('artifacts/raw.csv')
            # Define the Model Layers.
            # model = nn.Sequential(
            #     nn.Conv2d(16, 8, 3, stride=(2, 2), padding=(2, 2)),
            #     nn.ReLU(),
            #     nn.Conv2d(16, 8, 3, stride=(2, 2), padding=(2, 2)),
            #     nn.LogSoftmax(dim=1)
            # )
            model = nn.Sequential(
                nn.Linear(X_train.shape[1], 64).to('cuda'),
                nn.ReLU().to('cuda'),
                nn.Linear(64, df['password'].nunique()).to('cuda'),
                nn.LogSoftmax(dim=1).to('cuda')
            )

            # Define the Model's Loss.
            criterion = nn.NLLLoss().to('cuda')

            # Forward pass, log
            logps = model(X_train)

            # Calculate the loss with the logits and the labels
            loss = criterion(logps, y_train)

            loss.backward()

            # Optimizers need parameters to optimize and a learning rate
            optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

            epochs = 20
            for e in range(epochs):
                optimizer.zero_grad()
                output = model.forward(X_train)
                loss = criterion(output, y_train)
                loss.backward()
                optimizer.step()

            logging.info(f"Saving the Model.")

            torch.save(model.state_dict(),
                       self.model_trainer_config.model_file_path)
            logging.info(f"Successfully, Saved the Model.")

            with torch.no_grad():
                model.eval()
                log_ps = model(X_test)
                test_loss = criterion(log_ps, y_test)
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == y_test.view(*top_class.shape)
                test_accuracy = torch.mean(equals.float())
                print(f"Test Accuracy is: {test_accuracy}")

        except Exception as error:
            raise CustomException(error, sys)
