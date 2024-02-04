import os
import sys
import pandas as pd
from flask import Flask, render_template, request
from src.pipelines.predict_pipeline import PipelinePrediction, CustomDataClass
from src.exception import CustomException

app = Flask(__name__)

@app.route('/', methods=["GET", "POST"])
def home():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        custom_data_class = CustomDataClass(
            password=request.form.get('password')
        )
        features = custom_data_class.get_custom_data()
        print(features)
        predict_pipeline = PipelinePrediction()
        answer = predict_pipeline.predict_pipeline(features=features)
        return render_template('index.html', results = answer)

if __name__ == '__main__':
    app.run(debug=True)
