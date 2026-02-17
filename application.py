

from flask import Flask,request, render_template

import numpy as np
import pandas as pd
import pickle

from sklearn.preprocessing import StandardScaler
from src.logger import logger

from src.pipeline.predict_pipeline import CustomData,PredictPipeline

app = Flask(__name__)


@app.before_request
def log_request_info():
    # Skip logging for static files
    if request.path.startswith("/static/"):
        return
    logger.info(f"Request received | Method: {request.method} | URL: {request.url}")

@app.after_request
def log_response_info(response):
    if request.path.startswith("/static/"):
        return response
    logger.info(f"Response sent | Status: {response.status}")
    return response


@app.route('/')
def home():
    return render_template('home.html') 

@app.route('/predict',methods=['POST',"GET"])

def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')

    else:
        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('race_ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))
        )

        pred_df=data.get_data_as_dataframe()
        print(pred_df)

        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_df)
        return render_template("home.html",results=results[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0",port=5000)