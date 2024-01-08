# Supress Warnings
import warnings
warnings.filterwarnings('ignore')

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
from sklearn.metrics import f1_score, accuracy_score,classification_report,confusion_matrix,precision_score,recall_score


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

@dataclass
class DataCollectionConfig:
    collected_data_path_train_logs: str = os.path.join('artifacts','train_logs.csv')
    collected_data_path_train_scores: str = os.path.join('artifacts','train_scores.csv')
    
class DataCollection:
    def __init__(self):
        self.collection_config = DataCollectionConfig()
    
    
    def combine_two_datasets(self,dataset1,dataset2):
        data = pd.concat([dataset1,dataset2], axis=1)
        return data
    
    

    def initiate_data_collection(self):
        logging.info("Entered the data collection process")
        try:
            train_logs = pd.read_csv("train_logs.csv")
            train_scores=pd.read_csv("train_scores.csv")
            #train_data = self.combine_two_datasets(train_logs,train_scores)
            os.makedirs(os.path.dirname(self.collection_config.collected_data_path_train_logs),exist_ok=True)
            os.makedirs(os.path.dirname(self.collection_config.collected_data_path_train_scores),exist_ok=True)
            train_logs.to_csv(self.collection_config.collected_data_path_train_logs,index=False,header=True)
            train_scores.to_csv(self.collection_config.collected_data_path_train_scores,index=False,header=True)

            logging.info("Collected dataset as csv file")

        
            
            return  self.collection_config.collected_data_path_train_logs,self.collection_config.collected_data_path_train_scores

            
        except Exception as e:
            raise CustomException(e,sys)    
        
if __name__=="__main__":
    obj=DataCollection()
    train_logs,train_scores=obj.initiate_data_collection()
    
    data_ingestion = DataIngestion()
    train_data,test_data = data_ingestion.initiate_data_ingestion(train_logs,train_scores)
    
    
    data_transformation = DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)
    
    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr,test_arr))
    
    
    
