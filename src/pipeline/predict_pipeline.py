import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

import warnings
warnings.filterwarnings('ignore')
import json

# Visualization
import ipyleaflet
import matplotlib.pyplot as plt
from IPython.display import Image
import seaborn as sns

# Data Science
import numpy as np
import pandas as pd

# Feature Engineering
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Machine Learning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score,classification_report,confusion_matrix,precision_score,recall_score,r2_score
from datetime import datetime

# Others
import requests
#import rich.table
from itertools import cycle
from tqdm import tqdm
tqdm.pandas()
import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_ingestion import DataIngestionConfig
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.utils import transform

current_datetime = datetime.now().strftime('%Y%m%d_%H%M%S')



class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            #model_path = "artifacts\model.pkl"
            model_path = r"C:\Users\prath\Downloads\keystroke\artifacts\model.pkl"
            #preprocessor_path = "artifacts\preprocessor.pkl"
            preprocessor_path = r"C:\Users\prath\Downloads\keystroke\artifacts\preprocessor.pkl"
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print(features)
            features = transform(test_logs=features)
            id = features.iloc[0,0]
            print(".................")
            print(id)
            features.drop('id', axis=1)
            print(features)
            data_scaled = preprocessor.transform(features)
            score = model.predict(data_scaled)
            features['score'] = score
            output_folder = os.path.join('output',str(int(id)), current_datetime)
            os.makedirs(output_folder, exist_ok=True)
            csv_filename =  os.path.join(output_folder, f'{int(id)}_scoredata_{current_datetime}.csv')
            features.to_csv(csv_filename, index=False)
            logging.info('Prediction is done.')
            return score
        except Exception as e:
            raise CustomException(e,sys)
    
class CustomData:
    def __init__(self,jsonobject,jsonobject_dir,jsonobject_landmark1,jsonobject_landmark2,text,video,screen):
        self.jsonobject = jsonobject
        self.text = text
        self.jsonobject_dir = jsonobject_dir
        self.jsonobject_landmark1 = jsonobject_landmark1
        self.jsonobject_landmark2 = jsonobject_landmark2
        self.video = video
        self.screen = screen
        self.video.save('video.webm')
        self.screen.save('screen.webm')
        
    def get_data_as_data_frame(self): 
        try:
            json_data = json.loads(self.jsonobject)
            print(json_data)
            custom_data_input = pd.DataFrame(json_data) #columns=["id","event_id", "down_time", "up_time", "action_time","activity", "down_event","up_event","text_change", "cursor_position", "word_count"])
            id = custom_data_input.iloc[0,0]
            output_folder = os.path.join('output',str(id), current_datetime)
            os.makedirs(output_folder, exist_ok=True)
            csv_filename = os.path.join(output_folder, f'{id}_keylogdata_{current_datetime}.csv')
            custom_data_input.to_csv(csv_filename, index=False)
            logging.info("Keystroke data is collected")
            print(custom_data_input)
            
            jsondata_dir = json.loads(self.jsonobject_dir)
            custom_data_input_dir = pd.DataFrame(jsondata_dir) #columns = ["id", "event_id", "key", "xEye", "yEye", "xHead", "yHead",'actual_dir' ]
            csv_filename =  os.path.join(output_folder, f'{id}_directiondata_{current_datetime}.csv')
            custom_data_input_dir.to_csv(csv_filename, index=False,mode='w')
            custom_data_input_dir.to_csv("directions.csv", index=False, header = False, mode='a')
            logging.info(jsondata_dir)
            
            jsondata_landmark1 = json.loads(self.jsonobject_landmark1)
            custom_data_input_dir = pd.DataFrame(jsondata_landmark1) #columns = ["id", "event_id", "key", "xEye", "yEye", "xHead", "yHead",'actual_dir' ]
            csv_filename =  os.path.join(output_folder, f'{id}_landmark1_data_{current_datetime}.csv')
            custom_data_input_dir.to_csv(csv_filename, index=False,mode='w')
            custom_data_input_dir.to_csv("landmarks1.csv", index=False, header = False, mode='a')
            logging.info(jsondata_landmark1)
            
            jsondata_landmark2 = json.loads(self.jsonobject_landmark2)
            custom_data_input_dir = pd.DataFrame(jsondata_landmark2) #columns = ["id", "event_id", "key", "xEye", "yEye", "xHead", "yHead",'actual_dir' ]
            csv_filename =  os.path.join(output_folder, f'{id}_landmark2_data_{current_datetime}.csv')
            custom_data_input_dir.to_csv(csv_filename, index=False,mode='w')
            custom_data_input_dir.to_csv("landmarks2.csv", index=False, header = False, mode='a')
            logging.info(jsondata_landmark2)
            
            return custom_data_input
        except Exception as e:
            raise CustomException(e,sys)
        
#  // <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@3.0/dist/tf.min.js"></script>