import pickle
from flask import Flask,request,render_template,jsonify,redirect, url_for
import numpy as np
import pandas as pd
import base64
import os
import sys
from src.exception import CustomException
from src.logger import logging
import json
from datetime import datetime

from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application = Flask(__name__)
app=application
current_datetime = datetime.now().strftime('%Y%m%d_%H%M%S')

#http://localhost:5000/

def save_video_to_file(video_data):
    # Decode base64 video data
    video_binary = video_data.split(',')[1].encode('utf-8')
    video_bytes = base64.b64decode(video_binary)

    # Save video to a file (you can adjust the filename and extension)
    with open('captured_video.webm', 'wb') as video_file:
        video_file.write(video_bytes)

@app.route("/")
def index():
    
    return render_template('index1.html')

data = {}
@app.route("/home",methods=['GET','POST'])

def home():
    print(request.method)
    if request.method == "GET":
        return render_template('home2.html')
    else:
        Name = request.form.get('Name')
        reg_id = request.form.get('reg_id')
        email_id = request.form.get('email_id')
        screen_pixels = request.form.get('screen_pixels')
        screen_size = request.form.get('screen_size')
        cam_pixels = request.form.get('cam_pixels')
        frame_rate = request.form.get('frame_rate')
        data = {'reg_id':reg_id,'Name':Name,'email_id':email_id,'screen_pixels':screen_pixels, 'screen_size':screen_size, 'cam_pixels':cam_pixels,'frame_rate':frame_rate}
        data_df = {'reg_id':[reg_id],'Name':[Name],'email_id':[email_id],'screen_pixels':[screen_pixels], 'screen_size':[screen_size], 'cam_pixels':[cam_pixels],'frame_rate':[frame_rate]}
        data_df = pd.DataFrame(data_df)
        data_df.to_csv('output/data.csv',mode='a',index=False,header=None)
        return render_template('home2.html',data=data)
        #return redirect(url_for('success'))

@app.route("/success")
def success():
    return render_template('home2.html',data=data) 


@app.route("/predict_datapoint",methods=['GET','POST'])
def predict_datapoint():
    if request.method == "GET":
        logging.info('Score is predicted.')
        return render_template('home2.html')
    else:
        if len(json.loads(request.form.get('jsonobject'))) == 0 or len(json.loads(request.form.get('jsonobject_dir'))) == 0:
            return jsonify({'result':"0. Invalid attempt !!!!!!!"})
        logging.info(request.files['video'])
        logging.info(request.files['video'].stream.read())
        logging.info(request.files['screen'])
        logging.info(request.files['screen'].stream.read())
        data =  CustomData(
            jsonobject = request.form.get('jsonobject'),
            jsonobject_dir = request.form.get('jsonobject_dir'),
            jsonobject_landmark1 = request.form.get('jsonobject_landmark1'),
            jsonobject_landmark2 = request.form.get('jsonobject_landmark2'),
            text = request.form.get('text'),
            video =  request.files['video'],
            screen =  request.files['screen']
        )
        
        request.files['video'].save('video1.webm')
        request.files['screen'].save('screen1.webm')
        prediction_df = data.get_data_as_data_frame()
        
    
        print(prediction_df)
        
        logging.info(request.form.get('jsonobject_dir'))
        logging.info(request.form.get('jsonobject'))
        logging.info(request.form.get('text'))
        
        predict_pipeline = PredictPipeline()
        prediction = predict_pipeline.predict(prediction_df)
        logging.info('score is predicted.')
        #save_video_to_file(request.form.get('capturedVideo'))
        #return render_template('home1.html',data={'result' : str(prediction[0])})
        return jsonify({'result':str(prediction[0])})
        
if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True)