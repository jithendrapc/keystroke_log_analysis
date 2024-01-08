import pickle
from flask import Flask,request,render_template,jsonify
import numpy as np
import pandas as pd
import os
import sys
from src.exception import CustomException
from src.logger import logging

from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application = Flask(__name__)
app=application


@app.route("/")
def index():
    return render_template('index.html')

@app.route("/home")
def home():
    return render_template('home.html')



@app.route("/predict_datapoint",methods=['GET','POST'])
def predict_datapoint():
    if request.method == "GET":
        logging.info('Score is predicted.')
        return render_template('home.html')
    else:
        data =  CustomData(
            jsonobject = request.form.get('jsonobject')
        )
        prediction_df = data.get_data_as_data_frame()
        print(prediction_df)
        
        predict_pipeline = PredictPipeline()
        prediction = predict_pipeline.predict(prediction_df)
        logging.info('score is predicted.')
        return render_template('home.html',results=prediction[0])
        #return str(prediction[0])
        
if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True)