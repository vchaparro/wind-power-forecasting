import logging
from typing import Any, Dict
import numpy as np
import pandas as pd
import os
import mlflow
from mlflow import sklearn
from datetime import datetime


def train_model(
    train_x: pd.DataFrame, train_y: pd.DataFrame, parameters: Dict[str, Any]) -> np.ndarray:
    num_iter = parameters["example_num_train_iter"]
    lr = parameters["example_learning_rate"]
    X = train_x.values
    Y = train_y.values

    # Add bias to the features
    bias = np.ones((X.shape[0], 1))
    X = np.concatenate((bias, X), axis=1)
    mlflow.log_artifact(local_path=os.path.join("data", "01_raw", "iris.csv"))
    weights = []

    # Train one model for each class in Y
    for k in range(Y.shape[1]):
        # Initialise weights
        theta = np.zeros(X.shape[1])
        y = Y[:, k]
        for _ in range(num_iter):
            z = np.dot(X, theta)
            h = _sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            theta -= lr * gradient

    # Save the weights for each model
    weights.append(theta)

    # Return a joint multi-class model with weights for all classes
    model = np.vstack(weights).transpose()
    sklearn.log_model(sk_model=model, artifact_path="model")
    return model


def report_accuracy(predictions: np.ndarray, test_y: pd.DataFrame) -> None:
    # Get true class index
    target = np.argmax(test_y.values, axis=1)
    # Calculate accuracy of predictions
    accuracy = np.sum(predictions == target) / target.shape[0]
    # Log the accuracy of the model
    log = logging.getLogger(__name__)
    log.info("Model accuracy on test set: {0:.2f}%".format(accuracy * 100))
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_param("time of prediction", str(datetime.now()))
    mlflow.set_tag("Population", 2019)