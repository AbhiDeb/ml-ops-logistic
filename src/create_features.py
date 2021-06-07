# Create auto features in the training and validation set

from autofeat import AutoFeatClassifier
import os
import warnings
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from get_data import read_params
from urllib.parse import urlparse
import argparse
import joblib
import json
import mlflow

def create_auto_features(config_path):
    config = read_params(config_path)
    test_data_path = config["split_data"]["test_path"] 
    train_data_path = config["split_data"]["train_path"]
    
    train = pd.read_csv(train_data_path, sep=",")
    test = pd.read_csv(test_data_path, sep=",")
    
    target = [config["base"]["target_col"]]
    
    train_y = train[target]
    test_y = test[target]

    train_x = train.drop(target, axis=1)
    test_x = test.drop(target, axis=1)
    
    model = AutoFeatClassifier()
    
    train_x = model.fit_transform(train_x.to_numpy(), train_y.to_numpy().flatten())
    test_x = model.transform(test_x.to_numpy())
    
    pd.concat([train_x,train_y], axis=1).to_csv(train_data_path, sep=",", index=False, encoding="utf-8")
    pd.concat([test_x ,test_y ], axis=1).to_csv(test_data_path , sep=",", index=False, encoding="utf-8")



if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    create_auto_features(config_path=parsed_args.config)