import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from tqdm import tqdm

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts','train.csv')
    test_data_path: str = os.path.join('artifacts','test.csv')
    raw_data_path: str = os.path.join('artifacts','data.csv')
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
    def preprocess(self,train_logs):
        grouped = train_logs.groupby('id')
        train = pd.DataFrame()
        for group_name, group_df in tqdm(grouped):
            #print(f"Group: {group_name}")

            #production rate
            counts = group_df['activity'].value_counts()
            counts = counts[0]-counts[1]
            total_characters_process = group_df[group_df['activity'] == 'Input']['text_change'].str.len().sum()
            total_time_taken_process_minutes = (group_df['up_time'].iloc[-1] - group_df['down_time'].iloc[0]) / (1000*60)
            production_rate_process = total_characters_process / total_time_taken_process_minutes
            train.loc[group_name,'production_rate_process '] = production_rate_process 
            production_rate_product = counts / total_time_taken_process_minutes
            train.loc[group_name,'production_rate_product'] = production_rate_product
            production_rate_word_count = group_df['word_count'].iloc[-1] / total_time_taken_process_minutes
            train.loc[group_name,'production_rate_word_count'] = production_rate_word_count
    
    
            #pause length
            threshold = 2000
            group_df['IKI'] = group_df['down_time'].diff()
            num_of_pauses = (group_df['IKI']>threshold).sum()
            total_pause_time = group_df[group_df['IKI']>2000]['IKI'].sum()/(1000*60)
            proportion_of_pause_time = (total_pause_time/total_time_taken_process_minutes)*100
            mean_pause_duration = group_df['IKI'][group_df['IKI'] > threshold].mean()
            train.loc[group_name,'total_pause_time'] = total_pause_time
            train.loc[group_name,'proportion_of_pause_time'] = proportion_of_pause_time
            train.loc[group_name,'mean_pause_duration'] = mean_pause_duration
    
             #Revision
            mask = (group_df['activity'] == 'Remove/Cut').astype(int)
            groups = (mask != mask.shift()).cumsum()
            num_sequential_deletions = mask.groupby(groups).sum()
            num_sequential_deletions = (num_sequential_deletions>0).sum()
    
            mask = (group_df['activity'] == 'Input').astype(int)
            groups = (mask != mask.shift()).cumsum()
            num_sequential_insertions = mask.groupby(groups).sum()
            num_sequential_insertions = (num_sequential_insertions>0).sum()
    
            deletions = group_df[group_df['activity'] == 'Remove/Cut']
            insertions = group_df[group_df['activity'] == 'Input']
            total_chars_deletions = deletions['text_change'].str.len().sum()
            total_chars_insertions = insertions['text_change'].str.len().sum()

            # Calculate proportion of deletions and insertions (as % of total writing time)
            prop_deletions = (total_chars_deletions / total_time_taken_process_minutes) * 100
            prop_insertions = (total_chars_insertions / total_time_taken_process_minutes) * 100

            # Calculate product vs. process ratio
            product_vs_process_ratio = total_chars_insertions / (total_chars_insertions + total_chars_deletions)

            # Calculate number/length of revisions at the point of inscription and after transcription
            inscriptions = group_df[group_df['activity'] == 'Input']
            revisions_at_inscription = inscriptions[inscriptions.duplicated(subset='down_time', keep='last')]
            revisions_after_transcription = revisions_at_inscription[revisions_at_inscription['down_time'] != revisions_at_inscription['down_time'].shift()]

            # Calculate number of immediate revisions and distant revisions
            cursor_positions = group_df['cursor_position'].astype(str)
            immediate_revisions = revisions_at_inscription[cursor_positions == cursor_positions.shift()]
            distant_revisions = revisions_at_inscription[cursor_positions != cursor_positions.shift()]
    
            train.loc[group_name,'num_sequential_deletions'] = num_sequential_deletions
            train.loc[group_name,'num_sequential_insertions'] = num_sequential_insertions
            train.loc[group_name,'total_chars_insertions'] = total_chars_insertions
            train.loc[group_name,'total_chars_deletions'] = total_chars_deletions
            train.loc[group_name,'prop_deletions'] = prop_deletions
            train.loc[group_name,'prop_insertions'] = prop_insertions
            train.loc[group_name,'product_vs_process_ratio'] = product_vs_process_ratio
            train.loc[group_name,'revisions_at_inscription'] = len(revisions_at_inscription)
            train.loc[group_name,'revisions_after_transcription'] = len(revisions_after_transcription)
            train.loc[group_name,'immediate_revisions'] = len(immediate_revisions)
            train.loc[group_name,'distant_revisions'] = len(distant_revisions)
    
            #Bursts
            pauses = (group_df['IKI'] > threshold).astype(int)
            revisions = ((group_df['activity'] == 'Remove/Cut') | (group_df['activity'] == 'Input')).astype(int)

            # Identify P-bursts and R-bursts based on pauses and revisions
            p_bursts = ((pauses == 1) & (revisions == 0)).astype(int)
            r_bursts = ((pauses == 1) | (revisions == 1)).astype(int)

            # Calculate number of P-bursts and R-bursts
            num_p_bursts = p_bursts.sum()
            num_r_bursts = r_bursts.sum()


            # Calculate proportion of P-bursts and R-bursts (as % of total writing time)
            prop_p_bursts = (num_p_bursts / total_time_taken_process_minutes) * 100
            prop_r_bursts = (num_r_bursts / total_time_taken_process_minutes) * 100

            # Calculate lengths of P-bursts and R-bursts (in characters)
            p_burst_lengths = group_df.loc[p_bursts.diff().ne(0) & p_bursts == 1, 'text_change'].apply(len)
            r_burst_lengths = group_df.loc[r_bursts.diff().ne(0) & r_bursts == 1, 'text_change'].apply(len)
    
    
            train.loc[group_name,'num_p_bursts'] = num_p_bursts
            train.loc[group_name,'num_r_bursts'] = num_r_bursts
            train.loc[group_name,'prop_p_bursts'] = prop_p_bursts
            train.loc[group_name,'prop_r_bursts'] = prop_r_bursts
            train.loc[group_name,'prop_p_bursts'] = p_burst_lengths.sum() if not p_burst_lengths.empty else 0
            train.loc[group_name,'prop_r_bursts'] = r_burst_lengths.sum() if not r_burst_lengths.empty else 0

            #Process variance
            # Calculate total writing time (in milliseconds)
            total_time_ms = group_df['up_time'].iloc[-1] - group_df['down_time'].iloc[0]
            # Set the number of time intervals (e.g., 5 or 10)
            num_intervals = 10
            # Calculate interval duration in milliseconds
            interval_duration = total_time_ms / num_intervals

            # Create intervals and assign each keystroke to an interval
            group_df['Interval'] = (group_df['down_time'] - group_df['down_time'].iloc[0]) // interval_duration

            # Group by interval and count characters produced in each interval
            characters_per_interval = group_df.groupby('Interval')['text_change'].apply(lambda x: len("".join(x)))

            # Calculate characters per minute for each interval
            characters_per_minute_per_interval = (characters_per_interval / (interval_duration / (1000 * 60))).fillna(0)

            # Calculate standard deviation of characters produced per interval
            process_variance = characters_per_minute_per_interval.std()

            train.loc[group_name,'process_variance'] = process_variance
            
            
        return train
        
        
        
        
        
        
        
        
        
        
        
    
    def initiate_data_ingestion(self,train_logs,train_scores):
        logging.info("Entered the data ingestion method or component")
        try:
            train_logs = pd.read_csv(train_logs)
            train_scores = pd.read_csv(train_scores,index_col='id')
            train_logs['activity'] = np.where((train_logs['activity'] == 'Remove/Cut') | (train_logs['activity'] =='Nonproduction')| (train_logs['activity'] =='Input') , train_logs['activity'] , 'Input')
            
            train = self.preprocess(train_logs)
            train = pd.concat([train,train_scores],axis=1)
            print(train.head(50))
            
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            os.makedirs(os.path.dirname(self.ingestion_config.test_data_path),exist_ok=True)
            
            train.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(train,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            
            logging.info("Ingestion of data is completed")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            raise CustomException(e,sys)    
        
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion(r'C:\Users\prath\Downloads\keystroke\artifacts\train_logs.csv',r'C:\Users\prath\Downloads\keystroke\artifacts\train_scores.csv')
    
    data_transformation = DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)
    
    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr,test_arr))
    
    
    
