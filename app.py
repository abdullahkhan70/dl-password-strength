import os
import sys
import pandas as pd
from src.pipelines.predict_pipeline import PipelinePrediction
from src.exception import CustomException

if __name__ == '__main__':
    try:
        pipeline_prediction = PipelinePrediction()
        pipeline_prediction.predict_pipeline("myPassword")
    except Exception as error:
        raise CustomException(error, sys)
