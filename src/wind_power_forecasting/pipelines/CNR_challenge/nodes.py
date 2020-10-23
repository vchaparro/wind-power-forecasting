import datetime as dt
import logging
import os
import pickle
import re
import time
from functools import wraps
from pathlib import Path
from typing import Callable, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from kedro.framework import context
from metpy import calc
from metpy.units import units
from operational_analysis.toolkits import filters, power_curve
from pyearth import Earth
from sklearn.feature_selection import (
    SelectFromModel,
    SelectKBest,
    mutual_info_regression,
)
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from tensorflow import keras

from wind_power_forecasting.pipelines.data_engineering.nodes import (
    _find_outliers,
    _get_wind_speed,
    _interpolate_missing_values,
    _plot_flagged_pc,
    _save_fig,
)


def get_data_by_wf(
    wf: str, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series,
) -> pd.DataFrame:
    """Get data filterd by Wind Farm (wf).

    Args:
        X: X_train_raw.
        **y: y_train_raw (optional)
        wf: Wind Farm identification.

    Returns:
        X, y data frames filtered by Wind Farm.
    """

    # Row selection by WF
    X_train = X_train[X_train["WF"] == wf]
    X_train["Time"] = pd.to_datetime(X_train["Time"], format="%d/%m/%Y %H:%M")

    X_test = X_test[X_test["WF"] == wf]
    X_test["Time"] = pd.to_datetime(X_test["Time"], format="%d/%m/%Y %H:%M")

    # Save observations identification
    ID_X = X_train["ID"]

    # selecting rows of y_train and y_test
    y_train = y_train["Production"]
    y_train = y_train.loc[ID_X.values - 1]

    return X_train, X_test, y_train


def add_new_cols(
    new_cols: list, X_train: pd.DataFrame, X_test: pd.DataFrame
) -> pd.DataFrame:
    """Adds new columns to a given data frame.

    Args:
        new_cols: List with the column names to be added.
        X: data frame that will be expanded with new_cols.

    Returns:
        X expanded with the new columns and the columns that
        contains missing values.

    """
    cols_train = X_train.columns[3:]
    cols_test = X_test.columns[3:-9]

    for col in new_cols:
        X_train[col] = np.nan
        X_test[col] = np.nan

    return X_train, X_test, cols_train, cols_test


def input_missing_values(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    cols_train: List,
    cols_test: List,
    cols_to_interpol: List,
) -> pd.DataFrame:
    """Impute missing values based on the gap time between forecasted timestamp and NWP run.

    Args:
        X: the data frame where the missing will be inputed.
        cols: columns with missig values due to daily frequency of NWP.
        cols_to_interpol: columns with missing values due to hourly frequency of NWP.

    Returns:
        X: the data frame with inputed missing values.

    """
    regex = r"NWP(?P<NWP>\d{1})_(?P<run>\d{2}h)_(?P<fc_day>D\W?\d?)_(?P<weather_var>\w{1,4})"
    p = re.compile(regex)

    NWP_met_vars_dict = {
        "1": ["U", "V", "T"],
        "2": ["U", "V"],
        "3": ["U", "V", "T"],
        "4": ["U", "V", "CLCT"],
    }

    # Input missing values in X_train
    for col in reversed(cols_train):
        m = p.match(col)
        col_name = (
            "NWP"
            + m.group("NWP")
            + "_"
            + m.group("run")
            + "_"
            + m.group("fc_day")
            + "_"
            + m.group("weather_var")
        )

        for key, value in NWP_met_vars_dict.items():
            for i in value:
                if m.group("NWP") == key and m.group("weather_var") == i:
                    X_train["NWP" + key + "_" + i] = X_train[
                        "NWP" + key + "_" + i
                    ].fillna(X_train[col_name])

    # Input missing values in X_test
    for col in reversed(cols_test):
        m = p.match(col)
        col_name = (
            "NWP"
            + m.group("NWP")
            + "_"
            + m.group("run")
            + "_"
            + m.group("fc_day")
            + "_"
            + m.group("weather_var")
        )

        for key, value in NWP_met_vars_dict.items():
            for i in value:
                if m.group("NWP") == key and m.group("weather_var") == i:
                    X_test["NWP" + key + "_" + i] = X_test[
                        "NWP" + key + "_" + i
                    ].fillna(X_test[col_name])

    # Interpolate missing values when required.
    _interpolate_missing_values(X_train, cols_to_interpol)
    _interpolate_missing_values(X_test, cols_to_interpol)

    return X_train, X_test


def select_best_NWP_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> pd.DataFrame:
    """Select the features of the best NWP.

    Args:
        X: features data frame.

    Returns:
        Data frame with the best NWP features.

    """

    # Select the best NWP predictions for weather predictors
    """X_train["U"] = (
        X_train.NWP1_U + X_train.NWP2_U + X_train.NWP3_U + X_train.NWP4_U
    ) / 4
    X_train["V"] = (
        X_train.NWP1_V + X_train.NWP2_V + X_train.NWP3_V + X_train.NWP4_V
    ) / 4
    X_train["T"] = (X_train.NWP1_T + X_train.NWP3_T) / 2
    X_train["CLCT"] = X_train.NWP4_CLCT

    X_test["U"] = (X_test.NWP1_U + X_test.NWP2_U + X_test.NWP3_U + X_test.NWP4_U) / 4
    X_test["V"] = (X_test.NWP1_V + X_test.NWP2_V + X_test.NWP3_V + X_test.NWP4_V) / 4
    X_test["T"] = (X_test.NWP1_T + X_test.NWP3_T) / 2
    X_test["CLCT"] = X_test.NWP4_CLCT

    # Select final features
    X_train = X_train[["ID", "Time", "U", "V", "T", "CLCT"]]
    X_test = X_test[["ID", "Time", "U", "V", "T", "CLCT"]]"""

    X_train["U"] = X_train.NWP1_U
    X_train["V"] = X_train.NWP1_V
    X_train["T"] = X_train.NWP1_T
    X_train["CLCT"] = X_train.NWP4_CLCT

    X_test["U"] = X_test.NWP1_U
    X_test["V"] = X_test.NWP1_V
    X_test["T"] = X_test.NWP1_T
    X_test["CLCT"] = X_test.NWP4_CLCT

    # Select final features
    X_train = X_train[["ID", "Time", "U", "V", "T", "CLCT"]]
    X_test = X_test[["ID", "Time", "U", "V", "T", "CLCT"]]

    return X_train, X_test


def clean_outliers(X: pd.DataFrame, y: pd.Series, wf: str, *args) -> tuple:
    """It removes the outliers and returned cleaned X, y.

    Args:
        X: the feature data frame.
        y: the target.

    Returns:
        Cleaned X, y.

    """

    # Find outliers
    outliers = _find_outliers(X, y, wf, *args)

    # Power curve data
    X["Production"] = y.to_list()
    X["wspeed"] = X.apply(_get_wind_speed, axis=1)
    X_ = X[["wspeed", "Production"]]

    # Remove outliers
    for value in outliers.values():
        X_.wspeed = X_.wspeed[(~value)]
        X_.Production = X_.Production[(~value)]

    # select no-outliers observations
    X_cleaned = X.loc[X["wspeed"].isin(X_.wspeed)]
    y_cleaned = X_cleaned["Production"]

    del X["Production"], X["wspeed"], X_
    del X_cleaned["wspeed"], X_cleaned["Production"]

    return X_cleaned, y_cleaned


def fix_negative_values(
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series
) -> Dict:
    """Replaces negative values of CLCT and power production by 0.

    Args:
        X: the data frame containing CLCT column.
        y: the target with Power values.

    Returns:
        None, it replaces the values inplace.

    """
    processed_data = {}
    X_train.loc[X_train["CLCT"] < 0, "CLCT"] = 0.0
    X_test.loc[X_test["CLCT"] < 0, "CLCT"] = 0.0

    processed_data["X_train"] = X_train
    processed_data["X_test"] = X_test
    processed_data["y_train"] = y_train

    return processed_data


def export_data(folder: str, WF: str, df_dict: Dict,) -> None:
    """Export data frames to csv.

    Args:
        folder: the folder where the csv files will be saved.
        WF: Wind Farm identification.
        df_dict: a dictionary with key, value pairs df name, df values.

    Returns:
        None.
    """
    os.makedirs(folder + WF, exist_ok=True)

    X_train = df_dict.get("X_train")
    X_test = df_dict.get("X_test")
    y_train = df_dict.get("y_train")

    X_train.to_csv(
        folder + "{}/{}.csv".format(WF, "X_train"),
        index=False,
        date_format="%d/%m/%Y %H:%M",
    )

    X_test.to_csv(
        folder + "{}/{}.csv".format(WF, "X_test"),
        index=False,
        date_format="%d/%m/%Y %H:%M",
    )
    y_train.to_csv(
        folder + "{}/{}.csv".format(WF, "y_train"), index=False, header=False
    )


#### Modeling nodes #####


def train_model(alg: str, wf: str) -> object:
    """Retrains selected model using k-fold cross
    validation.

    Args:
        alg: model to train.
        wf: wind farm identificator.

    Returns:
        The re-trained model ready for predictions.
    """

    # Load context and get model folder.
    ctx = context.load_context("../wind-power-forecasting")
    source_folder = ctx.params.get("folder").get("mdl")
    output_folder = ctx.params.get("folder").get("cnr").get("models")
    n_splits = ctx.params.get("n_splits")

    # Load training data.
    X_train = ctx.catalog.load("Xtrain_pped_{}".format(wf))
    y_train = ctx.catalog.load("ytrain_{}".format(wf))

    if alg == "MARS":
        # Load model
        with open(source_folder + "{0}/{1}.pickle".format(wf, alg), "rb") as file:
            model = pickle.load(file)

        # Re-train the model with ts cross validation
        model.fit(X_train, y_train)
        tscv = TimeSeriesSplit(n_splits)
        mars_cv = cross_validate(model, X_train, y_train, cv=tscv, scoring="r2")
        cv_mean_score = np.mean(mars_cv.get("test_score"))

        logger = logging.getLogger(__name__)
        logger.info("CV mean accuracy: {:.2f}%".format(cv_mean_score * 100))

        # Save model
        os.makedirs(output_folder + wf, exist_ok=True)
        with open(output_folder + "{0}/{1}.pickle".format(wf, alg), "wb") as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    elif alg == "KNN":
        # Load model
        with open(source_folder + "{0}/{1}.pickle".format(wf, alg), "rb") as file:
            model = pickle.load(file)

        # Re-train the model with ts cross validation
        model.fit(X_train, y_train)
        tscv = TimeSeriesSplit(n_splits)
        knn_cv = cross_validate(model, X_train, y_train, cv=tscv, scoring="r2")
        cv_mean_score = np.mean(knn_cv.get("test_score"))

        logger = logging.getLogger(__name__)
        logger.info("CV mean accuracy: {:.2f}%".format(cv_mean_score * 100))

        # Save model
        os.makedirs(output_folder + wf, exist_ok=True)
        with open(output_folder + "{0}/{1}.pickle".format(wf, alg), "wb") as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    elif alg == "RF":
        # Load model
        with open(source_folder + "{0}/{1}.pickle".format(wf, alg), "rb") as file:
            model = pickle.load(file)

        # Re-train the model with ts cross validation
        model.fit(X_train, y_train)
        tscv = TimeSeriesSplit(n_splits)
        rf_cv = cross_validate(model, X_train, y_train, cv=tscv, scoring="r2")
        cv_mean_score = np.mean(rf_cv.get("test_score"))

        logger = logging.getLogger(__name__)
        logger.info("CV mean accuracy: {:.2f}%".format(cv_mean_score * 100))

        # Save model
        os.makedirs(output_folder + wf, exist_ok=True)
        with open(output_folder + "{0}/{1}.pickle".format(wf, alg), "wb") as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    elif alg == "SVM":
        # Load model
        with open(source_folder + "{0}/{1}.pickle".format(wf, alg), "rb") as file:
            model = pickle.load(file)

        # Re-train the model with ts cross validation
        model.fit(X_train, y_train)
        tscv = TimeSeriesSplit(n_splits)
        svm_cv = cross_validate(model, X_train, y_train, cv=tscv, scoring="r2")
        cv_mean_score = np.mean(svm_cv.get("test_score"))

        logger = logging.getLogger(__name__)
        logger.info("CV mean accuracy: {:.2f}%".format(cv_mean_score * 100))

        # Save model
        os.makedirs(output_folder + wf, exist_ok=True)
        with open(output_folder + "{0}/{1}.pickle".format(wf, alg), "wb") as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    elif alg == "ANN":
        # Load model
        model = tf.keras.models.load_model(source_folder + "{0}/{1}.h5".format(wf, alg))
        print(model)
        input("...")

        # Re-train the model with ts cross validation
        model.fit(X_train, y_train)
        tscv = TimeSeriesSplit(n_splits)
        ann_cv = cross_validate(model, X_train, y_train, cv=tscv, scoring="r2")
        cv_mean_score = np.mean(ann_cv.get("test_score"))

        logger = logging.getLogger(__name__)
        logger.info("CV mean accuracy: {:.2f}%".format(cv_mean_score * 100))

        # Save model
        model.save(output_folder + "{0}/{1}.h5".format(wf, alg))

    return model


def predict(wf: str, model: object, output_folder: str, alg: str) -> np.ndarray:
    """Predicts energy power production using
    on testing data of CNR

    Args:
        wf: wind darm identificator.
        model: the re-trained model to use.

    Returns:
        Predicitons ready to be appended in submission
        file for CNR challenge.

    """
    # Load test data for the wind farm
    ctx = context.load_context("../wind-power-forecasting")
    X_test = ctx.catalog.load("Xtest_{}".format(wf))
    ID_test = X_test["ID"]
    X_test_pped = ctx.catalog.load("Xtest_pped_{}".format(wf))

    # Predict
    predictions = model.predict(X_test_pped)

    # Build prediction matrix (ID,Production)

    pred_matrix = np.stack(
        (np.array(ID_test).astype(int), predictions.reshape(-1)), axis=-1
    )
    df_pred = pd.DataFrame(
        data=pred_matrix.reshape(-1, 2), columns=["ID", "Production"]
    )

    # Fix negative values in predicted production
    df_pred.loc[df_pred["Production"] < 0, "Production"] = 0.0

    # Create submission file
    submission_df = pd.DataFrame([], columns=["ID", "Production"])

    if not os.path.isfile(output_folder + "submission.csv".format(alg)):
        submission_df = submission_df.append(df_pred, ignore_index=True)
        submission_df.to_csv(
            output_folder + "submission.csv".format(alg), index=False, sep=","
        )
    else:
        submission_df = submission_df.append(df_pred, ignore_index=True,)
        submission_df.to_csv(
            output_folder + "submission.csv".format(alg),
            index=False,
            sep=",",
            mode="a",
            header=False,
        )

    return df_pred
