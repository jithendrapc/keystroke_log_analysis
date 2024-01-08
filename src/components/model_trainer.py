import os
import sys
from dataclasses import dataclass
import warnings
warnings.filterwarnings("ignore")

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models


# Visualization
import ipyleaflet
import matplotlib.pyplot as plt
from IPython.display import Image
import seaborn as sns

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score,classification_report,confusion_matrix,precision_score,recall_score,r2_score

import requests
import rich.table
from itertools import cycle
from tqdm import tqdm
tqdm.pandas()
#from statistics import mean

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor
from lightgbm import LGBMRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import VotingRegressor

from sklearn.pipeline import Pipeline
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting train and test input data")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            model1 = RandomForestRegressor()
            model2 = LGBMRegressor()
            model3 = XGBRegressor()
            model4 = ExtraTreesRegressor()
            model5 = AdaBoostRegressor()
            model6 = GradientBoostingRegressor()
            model7 = SVR()
            model8 = KNeighborsRegressor()
            
            
            param_grid_model1 = {
            'n_estimators': [50, 100, 200],
            'criterion': ['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
            'max_depth': [None, 5, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'min_weight_fraction_leaf': [0.0, 0.1, 0.2],
            'max_features': ['auto', 'sqrt', 'log2', None],
            'max_leaf_nodes': [None, 5, 10, 20],
            'min_impurity_decrease': [0.0, 0.1, 0.2],
            'bootstrap': [True, False],
            'oob_score': [True, False],
            'n_jobs': [-1],
            'random_state': [None, 42],
        # 'verbose': [0, 1],
            'warm_start': [True, False],
            'ccp_alpha': [0.0, 0.1, 0.2],
            'max_samples': [None, 100, 200]
            }

            param_grid_model2 = {
                'boosting_type': ['gbdt', 'dart', 'rf'],
                'num_leaves': [31, 50, 100],
                'max_depth': [-1, 5, 10],
                'learning_rate': [0.1, 0.2, 0.3],
                'n_estimators': [100, 200, 300],
                'subsample_for_bin': [200000, 300000, 400000],
                'objective': [None, 'regression', 'binary', 'multiclass', 'lambdarank'],
                'class_weight': [None, 'balanced'],
                'min_split_gain': [0.0, 0.1, 0.2],
                'min_child_weight': [0.001, 0.002, 0.005],
                'min_child_samples': [20, 30, 40],
                'subsample': [1.0, 0.8, 0.9],
                'subsample_freq': [0, 5, 10],
                'colsample_bytree': [1.0, 0.8, 0.9],
                'reg_alpha': [0.0, 0.1, 0.2],
                'reg_lambda': [0.0, 0.1, 0.2],
                'random_state': [None, 42],
                'n_jobs': [-1],
                'importance_type': ['split', 'gain']
            }
            
            param_grid_model3 = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_child_weight': [1, 3, 5],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9]
                # Add more as needed
            }

            param_grid_model4 = {
                'n_estimators': [50, 100, 200],
                'criterion': ['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
                'max_depth': [None, 5, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'min_weight_fraction_leaf': [0.0, 0.1, 0.2],
                'max_features': ['auto', 'sqrt', 'log2', None],
                'max_leaf_nodes': [None, 5, 10, 20],
                'min_impurity_decrease': [0.0, 0.1, 0.2],
                'bootstrap': [True, False],
                'oob_score': [True, False],
                'n_jobs': [-1],
                'random_state': [None, 42],
            # 'verbose': [0, 1],
                'warm_start': [True, False],
                'ccp_alpha': [0.0, 0.1, 0.2],
                'max_samples': [None, 100, 200]
            }


            param_grid_model5 = {
                'estimator': [None, DecisionTreeRegressor()],
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.1, 0.5, 1.0, 1.5],
                'loss': ['linear', 'square', 'exponential'],
                'random_state': [None, 42]
            }
            
            param_grid_model6 = {
                'loss': ['squared_error', 'absolute_error', 'huber', 'quantile'],
                'learning_rate': [0.01, 0.1, 0.5],
                'n_estimators': [50, 100, 200],
                'subsample': [0.8, 1.0],  # Adjust values based on your preference
                'criterion': ['friedman_mse', 'squared_error'],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'min_weight_fraction_leaf': [0.0, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_impurity_decrease': [0.0, 0.1, 0.2],
                'init': [None, GradientBoostingRegressor()],  # You might need to adjust the initialization method
                'random_state': [None, 42],  # Change or add seed values as needed
                'max_features': [None, 'sqrt', 'log2', 0.5],  # Modify the feature options
                'alpha': [0.5, 0.9, 1.0],
            # 'verbose': [0, 1, 2],
                'max_leaf_nodes': [None, 10, 20],
                'warm_start': [True, False],
                'validation_fraction': [0.1, 0.2, 0.3],
                'n_iter_no_change': [None, 5, 10],
                'tol': [1e-4, 1e-3, 1e-2],
                'ccp_alpha': [0.0, 0.1, 0.2]
            }



            param_grid_model7 = {
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto']
                # Add more as needed
            }

            param_grid_model8 = {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                'leaf_size': [20, 30, 40],
                'p': [1, 2],  # Adjust based on your preference for Minkowski distance
                'metric': ['euclidean', 'manhattan', 'minkowski'],
                #'n_jobs': [-1, None]  # Use all processors or default value
            }
            
            
            
            models = [
                (model1, param_grid_model1),
                (model2, param_grid_model2),
                (model3, param_grid_model3),
                (model4, param_grid_model4),
                (model5, param_grid_model5),
                (model6, param_grid_model6),
                (model7, param_grid_model7),
                (model8, param_grid_model8)
            ]
                        
            
            model1 = RandomForestRegressor()
            model2 = LGBMRegressor()
            model3 = XGBRegressor()
            model4 = ExtraTreesRegressor()
            model5 = AdaBoostRegressor()
            model6 = GradientBoostingRegressor()
            model7 = SVR()
            model8 = KNeighborsRegressor()
            
            model1.set_params(**{'criterion':'poisson','max_depth':20,'max_features':'sqrt','min_samples_leaf':2,'min_samples_split':2,'n_estimators':100})
            model2.set_params(**{'colsample_bytree':0.8,'learning_rate':0.1,'max_depth':5,'min_child_samples':20,'n_estimators':50,'subsample':0.7})
            model3.set_params(**{'colsample_bytree':0.8,'learning_rate':0.1,'max_depth':3,'min_child_samples':5,'n_estimators':100,'subsample':0.7})
            model4.set_params(**{'criterion':'poisson','max_depth':20,'max_features':'log2','n_estimators':200})
            model5.set_params(**{'learning_rate':0.1,'n_estimators':50})
            model6.set_params(**{'learning_rate':0.1,'max_depth':3,'min_samples_leaf':4,'min_samples_split':5,'n_estimators':100})
            model8.set_params(**{'algorithm':'auto','n_neighbors':10,'p':1,'weights':'distance'})

            
            model1 = RandomForestRegressor()
            model2 = LGBMRegressor()
            model3 = XGBRegressor()
            model4 = ExtraTreesRegressor()
            model5 = AdaBoostRegressor()
            model6 = GradientBoostingRegressor()
            model7 = SVR()
            model8 = KNeighborsRegressor()
            
            
            

            
            models = {
              "Random Forest Regressor" : model1,
              "LGBM Regressor" : model2,
              "XGB Regressor" : model3,
              "Extra Trees Regressor" : model4,
              "AdaBoost Regressor" : model5,
              "Gradient Boosting Regressor" : model6,
              "SVR" : model7,
              "KNeighbors Regressor" : model8
  
                
            }
            
            params={
                "Random Forest Regressor": param_grid_model1,
                "LGBM Regressor": param_grid_model2,
                "XGB Regressor": param_grid_model3,
                "Extra Trees Regressor": param_grid_model4,
                "Adaboost Regressor": param_grid_model5,
                "Gradient Boosting Regressor": param_grid_model6,
                "SVR": param_grid_model7,
                "KNeighbors Regressor": param_grid_model8
                
            }
            
            #model_report:dict = evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,params = params)
            
            #best_model_score = max(sorted(model_report.values()))
            
            #best_model_name = list(model_report.keys())[
            #    list(model_report.values()).index(best_model_score)
            #]
            
            #best_model = models[best_model_name]
            vc = VotingRegressor(estimators=[
                ('1',model1),('2',model2),('3',model3),('4',model4),('5',model5),('6',model6),('7',model7),('8',model8)],verbose = True)
            best_model = vc.fit(X_train,y_train)
            predicted = best_model.predict(X_test)
            best_model_score = r2_score(y_test,predicted)
            
            #if best_model_score  < 0.5:
                #raise CustomException("No best model",sys)
            
            logging.info("Found best base model - {0}.".format('best_model'))
                 
            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            predicted = best_model.predict(X_test)
            acc_score = r2_score(y_test,predicted)
            
            return acc_score
            
        except Exception as e:
            raise CustomException(e,sys)
        
        
        