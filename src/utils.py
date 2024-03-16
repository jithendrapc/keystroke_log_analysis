import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import dill
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, accuracy_score,classification_report,confusion_matrix,precision_score,recall_score,r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
            
    except Exception as e:
        raise CustomException(e,sys)
    
def plot_confusion_matrix(true_value,predicted_value,title,labels):
    '''
    Plots a confusion matrix.
    Attributes:
    true_value - The ground truth value for comparision.
    predicted_value - The values predicted by the model.
    title - Title of the plot.
    labels - The x and y labels of the plot.
    '''
    cm = confusion_matrix(true_value,predicted_value)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax, cmap='Blues')
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title(title)
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)
    plt.show()

def transform(test_logs):
    grouped = test_logs.groupby('id')
    test = pd.DataFrame()
    for group_name, group_df in grouped:
        #print(f"Group: {group_name}")
        test.loc[group_name,'id'] = int(group_name)
        #production rate
        counts = group_df['activity'].value_counts()
        #print(counts.ndim)
        if counts.ndim > 1:
            counts = counts[0]-counts[1]
        else:
            counts = counts[0]
        total_characters_process = group_df['text_change'].str.len().sum()
        total_time_taken_process_minutes = (group_df['up_time'].iloc[-1] - group_df['down_time'].iloc[0]) / (1000*60)
        production_rate_process = total_characters_process / total_time_taken_process_minutes
        test.loc[group_name,'production_rate_process '] = production_rate_process 
        production_rate_product = counts / total_time_taken_process_minutes
        test.loc[group_name,'production_rate_product'] = production_rate_product
        production_rate_word_count = group_df['word_count'].iloc[-1] / total_time_taken_process_minutes
        test.loc[group_name,'production_rate_word_count'] = production_rate_word_count
        
        
        #pause length
        threshold = 2000
        group_df['IKI'] = group_df['down_time'].diff()
        num_of_pauses = (group_df['IKI']>threshold).sum()
        total_pause_time = group_df[group_df['IKI']>2000]['IKI'].sum()/(1000*60)
        proportion_of_pause_time = (total_pause_time/total_time_taken_process_minutes)*100
        mean_pause_duration = group_df['IKI'][group_df['IKI'] > threshold].mean()
        test.loc[group_name,'total_pause_time'] = total_pause_time
        test.loc[group_name,'proportion_of_pause_time'] = proportion_of_pause_time
        test.loc[group_name,'mean_pause_duration'] = mean_pause_duration
        
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
        
        test.loc[group_name,'num_sequential_deletions'] = num_sequential_deletions
        test.loc[group_name,'num_sequential_insertions'] = num_sequential_insertions
        test.loc[group_name,'total_chars_insertions'] = total_chars_insertions
        test.loc[group_name,'total_chars_deletions'] = total_chars_deletions
        test.loc[group_name,'prop_deletions'] = prop_deletions
        test.loc[group_name,'prop_insertions'] = prop_insertions
        test.loc[group_name,'product_vs_process_ratio'] = product_vs_process_ratio
        test.loc[group_name,'revisions_at_inscription'] = len(revisions_at_inscription)
        test.loc[group_name,'revisions_after_transcription'] = len(revisions_after_transcription)
        test.loc[group_name,'immediate_revisions'] = len(immediate_revisions)
        test.loc[group_name,'distant_revisions'] = len(distant_revisions)
        
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
        
        
        test.loc[group_name,'num_p_bursts'] = num_p_bursts
        test.loc[group_name,'num_r_bursts'] = num_r_bursts
        test.loc[group_name,'prop_p_bursts'] = prop_p_bursts
        test.loc[group_name,'prop_r_bursts'] = prop_r_bursts
        test.loc[group_name,'prop_p_bursts'] = p_burst_lengths.sum() if not p_burst_lengths.empty else 0
        test.loc[group_name,'prop_r_bursts'] = r_burst_lengths.sum() if not r_burst_lengths.empty else 0

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

        test.loc[group_name,'process_variance'] = process_variance
        
        print(test)
    
    return test
    
    
    
    

    
    
    
    
    
        
def evaluate_models(X_train,y_train,X_test,y_test,models,params):
    try:
        report={}
        
        for i in range(len(list(models))):
            model = list(models.values())[i]
            print(model)
            para=params[list(models.keys())[i]]
            grid_search_cv = GridSearchCV(model,para,cv=3)
            grid_search_cv.fit(X_train,y_train)
            model.set_params(**grid_search_cv.best_params_)
            model.fit(X_train,y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            train_model_score = r2_score(y_train_pred,y_train)
            test_model_score = r2_score(y_test_pred,y_test)
            report[list(models.keys())[i]] = test_model_score
            
        return report
    except Exception as e:
        raise CustomException(e,sys)
    
    
def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return dill.load(file_obj)
    
    except Exception as e:
        raise CustomException(e,sys)