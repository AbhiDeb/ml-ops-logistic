# load the train and test
# train algo
# save the metrices, params
import os
import warnings
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score,accuracy_score,recall_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from get_data import read_params
from urllib.parse import urlparse
import argparse
import joblib
import json
import mlflow

def eval_metrics(actual, pred):
    precision = precision_score(actual, pred)
    accuracy = accuracy_score(actual, pred)
    recall = recall_score(actual, pred)
    return precision, accuracy, recall

def train_and_evaluate(config_path):
    config = read_params(config_path)
    test_data_path = config["split_data"]["test_path"]
    train_data_path = config["split_data"]["train_path"]
    random_state = config["base"]["random_state"]
    model_dir = config["model_dir"]

    C = config["estimators"]["LogisticRegression"]["params"]["C"]
    penalty = config["estimators"]["LogisticRegression"]["params"]["penalty"]
    solver = config["estimators"]["LogisticRegression"]["params"]["solver"]

    target = [config["base"]["target_col"]]

    train = pd.read_csv(train_data_path, sep=",")
    test = pd.read_csv(test_data_path, sep=",")

    train_y = train[target]
    test_y = test[target]

    train_x = train.drop(target, axis=1)
    test_x = test.drop(target, axis=1)
    
    mms = MinMaxScaler()
    train_x = mms.fit_transform(train_x)
    test_x = mms.transform(test_x)

################### MLFLOW ###############################
    mlflow_config = config["mlflow_config"]
    remote_server_uri = mlflow_config["remote_server_uri"]

    mlflow.set_tracking_uri(remote_server_uri)

    mlflow.set_experiment(mlflow_config["experiment_name"])

    with mlflow.start_run(run_name=mlflow_config["run_name"]) as mlops_run:
        lr = LogisticRegression(
            C=C, 
            penalty=penalty,
            solver=solver,
            random_state=random_state)
        lr.fit(train_x, train_y.to_numpy().flatten())

        predicted_qualities = lr.predict(test_x)
        
        (precision, accuracy, recall) = eval_metrics(test_y, predicted_qualities)

        mlflow.log_param("C", C)
        mlflow.log_param("penalty", penalty)
        mlflow.log_param("solver", solver)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme

        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
                lr, 
                "model", 
                registered_model_name=mlflow_config["registered_model_name"])
        else:
            mlflow.sklearn.load_model(lr, "model")

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)