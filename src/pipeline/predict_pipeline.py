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

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            model_path = "artifacts\model.pkl"
            preprocessor_path = "artifacts\preprocessor.pkl"
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print(features)
            features = transform(test_logs=features)
            print(features)
            data_scaled = preprocessor.transform(features)
            score = model.predict(data_scaled)
            logging.info('Prediction is done.')
            return score
        except Exception as e:
            raise CustomException(e,sys)
    
class CustomData:
    def __init__(self,jsonobject):
        self.jsonobject = jsonobject
    
    def get_data_as_data_frame(self):
        try:
            json_data = json.loads(self.jsonobject)
            print(json_data)
            custom_data_input = pd.DataFrame(json_data) #columns=["id","event_id", "down_time", "up_time", "action_time","activity", "down_event","up_event","text_change", "cursor_position", "word_count"])
            logging.info("Keystroke data is collected")
            print(custom_data_input)
            return custom_data_input
        except Exception as e:
            raise CustomException(e,sys)