import datetime as dt
import logging
import os
import pickle
import re
import time
from functools import wraps
from pathlib import Path
from typing import Callable, Dict, List


import cufflinks as cf
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import tensorflow as tf
from kedro.framework import context
from mlflow import log_artifact, log_metric, log_param
from pyearth import Earth
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import (
    SelectFromModel,
    SelectKBest,
    mutual_info_regression,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics.scorer import make_scorer
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.neighbors import DistanceMetric, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, MinMaxScaler, StandardScaler
from sklearn.svm import SVR
from tensorflow import keras
from yellowbrick.model_selection import RFECV, CVScores, LearningCurve, ValidationCurve
from yellowbrick.regressor import PredictionError, ResidualsPlot

from wind_power_forecasting.nodes import data_transformation as dtr
from wind_power_forecasting.nodes import metric
from wind_power_forecasting.pipelines.feature_engineering.nodes import (
    feature_engineering as fe,
)

setattr(plotly.offline, "__PLOTLY_OFFLINE_INITIALIZED", True)
cf.set_config_file(offline=True)

# Cross validation strategy taken from
# https://hub.packtpub.com/cross-validation-strategies-for-time-series-forecasting-tutorial/


class BlockedTimeSeriesSplit:
    def __init__(self, n_splits):
        self.n_splits = n_splits

    def get_n_splits(self, X, y, groups):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        k_fold_size = n_samples // self.n_splits
        indices = np.arange(n_samples)

        margin = 0
        for i in range(self.n_splits):
            start = i * k_fold_size
            stop = start + k_fold_size
            mid = int(0.8 * (stop - start)) + start
            yield indices[start:mid], indices[mid + margin : stop]


def _log_gcv_scores(gcv: object, scoring: Dict, alg: str) -> Dict:
    results = gcv.cv_results_
    logger = logging.getLogger(__name__)

    for scorer in sorted(scoring):
        best_index = np.nonzero(results["rank_test_%s" % scorer] == 1)[0][0]
        best_CV = results["mean_test_%s" % scorer][best_index]
        best_CV_std = results["std_test_%s" % scorer][best_index]

        if (scorer == "RMSE") or (scorer == "MAE") or (scorer == "CAPE"):
            logger.info(
                "Best CV {0}:{1:.2f}, std:{2:.2f}".format(scorer, -best_CV, best_CV_std)
            )
        else:
            logger.info(
                "Best CV {0}:{1:.2f}, std:{2:.2f}".format(scorer, best_CV, best_CV_std)
            )

    # Log best hyperparameters found
    logger.info(
        "Best hyperparameters found for {0} : {1}".format(alg, gcv.best_params_)
    )


def _eval_metrics(actual, pred):

    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    cape = metric.get_cape(actual, pred)

    return rmse, mae, r2, cape


def _get_data_by_WF(wf: str):
    # Load context
    ctx = context.load_context("../wind-power-forecasting/")

    # Load data and feature names for the selected WF
    if wf == "WF1":
        X_train = ctx.catalog.load("X_train_pped_WF1")
        y_train = ctx.catalog.load("y_train_WF1")
        y_train = y_train[0]
        X_test = ctx.catalog.load("X_test_pped_WF1")
        y_test = ctx.catalog.load("y_test_WF1")
        y_test = y_test[0]
        feature_names = ctx.catalog.load("feature_names_WF1")
    elif wf == "WF2":
        X_train = ctx.catalog.load("X_train_pped_WF2")
        y_train = ctx.catalog.load("y_train_WF2")
        y_train = y_train[0]
        X_test = ctx.catalog.load("X_test_pped_WF2")
        y_test = ctx.catalog.load("y_test_WF2")
        y_test = y_test[0]
        feature_names = ctx.catalog.load("feature_names_WF2")
    elif wf == "WF3":
        X_train = ctx.catalog.load("X_train_pped_WF3")
        y_train = ctx.catalog.load("y_train_WF3")
        y_train = y_train[0]
        X_test = ctx.catalog.load("X_test_pped_WF3")
        y_test = ctx.catalog.load("y_test_WF3")
        y_test = y_test[0]
        feature_names = ctx.catalog.load("feature_names_WF3")
    elif wf == "WF4":
        X_train = ctx.catalog.load("X_train_pped_WF4")
        y_train = ctx.catalog.load("y_train_WF4")
        y_train = y_train[0]
        X_test = ctx.catalog.load("X_test_pped_WF4")
        y_test = ctx.catalog.load("y_test_WF4")
        y_test = y_test[0]
        feature_names = ctx.catalog.load("feature_names_WF4")
    elif wf == "WF5":
        X_train = ctx.catalog.load("X_train_pped_WF5")
        y_train = ctx.catalog.load("y_train_WF5")
        y_train = y_train[0]
        X_test = ctx.catalog.load("X_test_pped_WF5")
        y_test = ctx.catalog.load("y_test_WF5")
        y_test = y_test[0]
        feature_names = ctx.catalog.load("feature_names_WF5")
    elif wf == "WF6":
        X_train = ctx.catalog.load("X_train_pped_WF6")
        y_train = ctx.catalog.load("y_train_WF6")
        y_train = y_train[0]
        X_test = ctx.catalog.load("X_test_pped_WF6")
        y_test = ctx.catalog.load("y_test_WF6")
        y_test = y_test[0]
        feature_names = ctx.catalog.load("feature_names_WF6")

    return X_train, y_train, X_test, y_test, feature_names


def _save_model(folder: str, model, wf: str, alg: str, predictions: np.ndarray) -> None:
    """Saves the trained model and predictions.

    Args:
        folder: the folder where the pickle objects will be saved.
        WF: Wind Farm identification.
        model: the model object.
        alg: the algorithm used to create the model.
        predictions: predictions values got with the algorithm.

    Returns:
        None.
    """
    ctx = context.load_context("../wind-power-forecasting")
    pred_folder = ctx.params.get("folder").get("mout")
    os.makedirs(folder + wf, exist_ok=True)
    os.makedirs(pred_folder + wf, exist_ok=True)

    with open(folder + "{0}/{1}.pickle".format(wf, alg), "wb") as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(
        pred_folder + "{0}/{1}_predictions.pickle".format(wf, alg), "wb"
    ) as handle:
        pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _build_mars(
    X_train: np.ndarray,
    y_train: pd.Series,
    n_splits: int,
    find_max_kbests: bool,
    max_k_bests: int,
    scoring: Dict,
    refit: str,
    transform_target: bool,
) -> object:
    """Trains, cross-validates and tunes a
    Multivariate Adaptative Regression Splines (MARS) algorithm.

    Args:
        X_train: features training values.
        y_trai: target traning values.
        n_splits: number of splits for CV.
        find_max_kbests: whether to search or not for the maximum number of features to use.
        max_k_bests: maximum k bests features to use in feature selection
        scoring: dictionary with the scores for CV multiscoring.
        refit: the metric used to fit data in CV.
        transform_target: whether to apply or not a target transformation.

    Returns:
        An Grid Search CV scikit learn object.

    """
    ctx = context.load_context("../wind-power-forecasting")

    if find_max_kbests:
        k_bests = list(range(1, max_k_bests + 1))
    else:
        k_bests = [max_k_bests]

    selec_k_best = SelectKBest(mutual_info_regression, k=1)

    if transform_target:
        mars = TransformedTargetRegressor(
            regressor=Earth(feature_importance_type="gcv"),
            transformer=PowerTransformer(method="yeo-johnson", standardize=True),
            check_inverse=False,
        )
        pipeline = Pipeline(
            [
                (
                    "powertransformer",
                    PowerTransformer(method="yeo-johnson", standardize=True),
                ),
                ("univariate_sel", selec_k_best),
                ("mars", mars),
            ]
        )
        param_grid = {
            "mars__regressor__max_degree": ctx.params.get("mars_hypms").get(
                "max_degree"
            ),
            "mars__regressor__allow_linear": ctx.params.get("mars_hypms").get(
                "allow_linear"
            ),
            "mars__regressor__penalty": ctx.params.get("mars_hypms").get("penalty"),
            "univariate_sel__k": k_bests,
        }

    else:
        mars = Earth(feature_importance_type="gcv")
        pipeline = Pipeline(
            [
                (
                    "powertransformer",
                    PowerTransformer(method="yeo-johnson", standardize=True),
                ),
                ("univariate_sel", selec_k_best),
                ("mars", mars),
            ]
        )
        param_grid = {
            "mars__max_degree": ctx.params.get("mars_hypms").get("max_degree"),
            "mars__allow_linear": ctx.params.get("mars_hypms").get("allow_linear"),
            "mars__penalty": ctx.params.get("mars_hypms").get("penalty"),
            "univariate_sel__k": k_bests,
        }

    # Cross validation and hyper-parameter tunning with grid search
    n_splits = n_splits
    refit = refit
    tscv = TimeSeriesSplit(n_splits)
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=tscv, scoring=scoring, refit=refit, n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    best_mars = grid_search.best_estimator_

    _log_gcv_scores(grid_search, scoring, "MARS")

    return best_mars


def _test_best_mars(
    wf,
    best_mars,
    alg,
    X_train,
    y_train,
    X_test,
    y_test,
    feature_names,
    output_folder,
    transform_target,
):

    # Feature selection with the best k value obtanined in Grid Search
    selec_k_best = SelectKBest(
        mutual_info_regression, k=best_mars.get_params().get("univariate_sel__k")
    )
    selec_k_best.fit(X_train, y_train)
    mask = selec_k_best.get_support()  # list of booleans
    selected_feat = []

    for bool, feature in zip(mask, feature_names):
        if bool:
            selected_feat.append(feature)

    # Predict
    predictions = best_mars.predict(X_test)

    # fix negative values in target predictions
    predictions[predictions < 0] = 0.0

    # get metrics
    (rmse, mae, r2, cape) = _eval_metrics(y_test, predictions)

    # Print metrics
    logger = logging.getLogger(__name__)
    logger.info("Test CAPE: {:.2f}".format(cape))
    logger.info("Test MAE: {:.2f}".format(mae))
    logger.info("Test R2: {:.2f}".format(r2))
    logger.info("Test rmse: {:.2f}".format(rmse))

    ### MLFlow logging ####

    mlflow.log_param("selected_features", selected_feat)
    mlflow.log_param("k_best_features", best_mars.get_params().get("univariate_sel__k"))

    if transform_target:
        mlflow.log_param(
            "max_degree", best_mars.get_params().get("mars__regressor__max_degree")
        )
        mlflow.log_param(
            "penalty", best_mars.get_params().get("mars__regressor__penalty")
        )
        mlflow.log_param(
            "allow_linear", best_mars.get_params().get("mars__regressor__allow_linear")
        )
    else:
        mlflow.log_param("max_degree", best_mars.get_params().get("mars__max_degree"))
        mlflow.log_param("penalty", best_mars.get_params().get("mars__penalty"))
        mlflow.log_param(
            "allow_linear", best_mars.get_params().get("mars__allow_linear")
        )

    # metrics
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("cape", cape)

    # artifacts
    mlflow.sklearn.log_model(best_mars, "MARS")

    # save model
    _save_model(output_folder, best_mars, wf, alg, predictions)

    return best_mars, X_train, X_test, predictions


def _build_knn(
    X_train: np.ndarray,
    y_train: pd.Series,
    n_splits: int,
    find_max_kbests: bool,
    max_k_bests: int,
    scoring: Dict,
    refit: str,
    transform_target: bool,
) -> object:
    """Trains, cross-validates and tunes a k-NN algorithm.

    Args:
        X_train: features training values.
        y_trai: target traning values.
        n_splits: number of splits for CV.
        find_max_kbests: whether to search or not for the maximum number of features to use.
        max_k_bests: maximum k bests features to use in feature selection
        scoring: dictionary with the scores for CV multiscoring.
        refit: the metric used to fit data in CV.
        transform_target: whether to apply or not a target transformat

    Returns:
        An Grid Search CV scikit learn object.

    """
    selec_k_best = SelectKBest(mutual_info_regression, k=1)

    if find_max_kbests:
        k_bests = list(range(1, max_k_bests + 1))
    else:
        k_bests = [max_k_bests]

    # Load knn hyperparams
    ctx = context.load_context("../wind-power-forecasting")
    n_neighbors = list(range(1, ctx.params.get("knn_hypms").get("max_n_neighbors"), 2))

    if transform_target:
        knn = TransformedTargetRegressor(
            regressor=KNeighborsRegressor(),
            transformer=PowerTransformer(method="yeo-johnson", standardize=True),
            check_inverse=False,
        )
        pipeline = Pipeline(
            [
                (
                    "powertransformer",
                    PowerTransformer(method="yeo-johnson", standardize=True),
                ),
                ("univariate_sel", selec_k_best),
                ("knn", knn),
            ]
        )
        param_grid = {
            "knn__regressor__n_neighbors": n_neighbors,
            "knn__regressor__algorithm": ctx.params.get("knn_hypms").get("algorithm"),
            "knn__regressor__weights": ctx.params.get("knn_hypms").get("weights"),
            "knn__regressor__metric": ctx.params.get("knn_hypms").get("metric"),
            "univariate_sel__k": k_bests,
        }
    else:
        knn = KNeighborsRegressor()
        pipeline = Pipeline(
            [
                (
                    "powertransformer",
                    PowerTransformer(method="yeo-johnson", standardize=True),
                ),
                ("univariate_sel", selec_k_best),
                ("knn", knn),
            ]
        )
        param_grid = {
            "knn__n_neighbors": n_neighbors,
            "knn__algorithm": ctx.params.get("knn_hypms").get("algorithm"),
            "knn__weights": ctx.params.get("knn_hypms").get("weights"),
            "knn__metric": ctx.params.get("knn_hypms").get("metric"),
            "univariate_sel__k": k_bests,
        }

    # Cross validation and hyper-parameter tunning with grid search
    n_splits = n_splits
    refit = refit
    tscv = TimeSeriesSplit(n_splits)
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=tscv, scoring=scoring, refit=refit, n_jobs=-1
    )

    grid_search.fit(X_train, y_train)
    best_knn = grid_search.best_estimator_
    _log_gcv_scores(grid_search, scoring, "KNN")

    return best_knn


def _test_best_knn(
    wf,
    best_knn,
    alg,
    X_train,
    y_train,
    X_test,
    y_test,
    feature_names,
    output_folder,
    transform_target,
):

    # Feature selection with the best k value obtanined in Grid Search
    selec_k_best = SelectKBest(
        mutual_info_regression, k=best_knn.get_params().get("univariate_sel__k")
    )
    selec_k_best.fit(X_train, y_train)
    mask = selec_k_best.get_support()  # list of booleans
    selected_feat = []

    for bool, feature in zip(mask, feature_names):
        if bool:
            selected_feat.append(feature)

    predictions = best_knn.predict(X_test)

    # fix negative values in target predictions
    predictions[predictions < 0] = 0.0

    # get metrics
    (rmse, mae, r2, cape) = _eval_metrics(y_test, predictions)

    # Printmetrics
    logger = logging.getLogger(__name__)
    logger.info("Test CAPE: {:.2f}".format(cape))
    logger.info("Test MAE: {:.2f}".format(mae))
    logger.info("Test R2: {:.2f}".format(r2))
    logger.info("Test rmse: {:.2f}".format(rmse))

    ### MLFlow logging ####

    # pre-processing
    mlflow.log_param("selected_features", selected_feat)
    mlflow.log_param("k_best_features", best_knn.get_params().get("univariate_sel__k"))

    # grid search parameters
    if transform_target:
        mlflow.log_param(
            "n_neighbors", best_knn.get_params().get("knn__regressor__n_neighbors")
        )
        mlflow.log_param(
            "algorithm", best_knn.get_params().get("knn__regressor__algorithm")
        )
        mlflow.log_param(
            "weights", best_knn.get_params().get("knn__regressor__weights")
        )
        mlflow.log_param("metric", best_knn.get_params().get("knn__regressor__metric"))

    else:
        mlflow.log_param("n_neighbors", best_knn.get_params().get("knn__n_neighbors"))
        mlflow.log_param("algorithm", best_knn.get_params().get("knn__algorithm"))
        mlflow.log_param("weights", best_knn.get_params().get("knn__weights"))
        mlflow.log_param("metric", best_knn.get_params().get("knn__metric"))

    # metrics
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("cape", cape)

    # artifacts
    mlflow.sklearn.log_model(best_knn, "KNN")

    # save model
    _save_model(output_folder, best_knn, wf, alg, predictions)

    return best_knn, X_train, X_test, predictions


def _build_svm(
    X_train: np.ndarray,
    y_train: pd.Series,
    n_splits: int,
    find_max_kbests: bool,
    max_k_bests: int,
    scoring: Dict,
    refit: str,
    transform_target: bool,
) -> object:
    """Trains, cross-validates and tunes a SVM algorithm.

    Args:
        X_train: features training values.
        y_trai: target traning values.
        n_splits: number of splits for CV.
        k_bests: list with the k bests feature to use in feature selection
        scoring: dictionary with the scores for CV multiscoring.
        refit: the metric used to fit data in CV.
        transform_target: whether to transform or not the target.

    Returns:
        An Grid Search CV scikit learn object.

    """
    ctx = context.load_context("../wind-power-forecasting")

    if find_max_kbests:
        k_bests = list(range(1, max_k_bests + 1))
    else:
        k_bests = [max_k_bests]

    selec_k_best = SelectKBest(mutual_info_regression, k=1)

    if transform_target:
        svm = TransformedTargetRegressor(
            regressor=SVR(), transformer=StandardScaler(), check_inverse=False,
        )
        pipeline = Pipeline(
            [
                (
                    "powertransformer",
                    PowerTransformer(method="yeo-johnson", standardize=True),
                ),
                ("univariate_sel", selec_k_best),
                ("svm", svm),
            ]
        )
        param_grid = {
            "svm__regressor__kernel": ctx.params.get("svm_hypms").get("kernel"),
            "svm__regressor__C": ctx.params.get("svm_hypms").get("C"),
            "svm__regressor__gamma": ctx.params.get("svm_hypms").get("gamma"),
            "svm__regressor__epsilon": ctx.params.get("svm_hypms").get("epsilon"),
            "univariate_sel__k": k_bests,
        }
    else:
        svm = SVR()
        pipeline = Pipeline(
            [
                (
                    "powertransformer",
                    PowerTransformer(method="yeo-johnson", standardize=True),
                ),
                ("univariate_sel", selec_k_best),
                ("svm", svm),
            ]
        )
        param_grid = {
            "svm__kernel": ctx.params.get("svm_hypms").get("kernel"),
            "svm__C": ctx.params.get("svm_hypms").get("C"),
            "svm__gamma": ctx.params.get("svm_hypms").get("gamma"),
            "svm__epsilon": ctx.params.get("svm_hypms").get("epsilon"),
            "univariate_sel__k": k_bests,
        }

    # Cross validation and hyper-parameter tunning with grid search
    n_splits = n_splits
    refit = refit
    tscv = TimeSeriesSplit(n_splits)
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=tscv, scoring=scoring, refit=refit, n_jobs=-1
    )

    grid_search.fit(X_train, y_train)
    best_svm = grid_search.best_estimator_
    _log_gcv_scores(grid_search, scoring, "SVM")

    return best_svm


def _test_best_svm(
    wf,
    best_svm,
    alg,
    X_train,
    y_train,
    X_test,
    y_test,
    feature_names,
    output_folder,
    transform_target,
):

    # Feature selection with the best k value obtanined in Grid Search
    selec_k_best = SelectKBest(
        mutual_info_regression, k=best_svm.get_params().get("univariate_sel__k")
    )
    selec_k_best.fit(X_train, y_train)
    mask = selec_k_best.get_support()  # list of booleans
    selected_feat = []

    for bool, feature in zip(mask, feature_names):
        if bool:
            selected_feat.append(feature)

    predictions = best_svm.predict(X_test)

    # fix negative values in target predictions
    predictions[predictions < 0] = 0.0

    # get metrics
    (rmse, mae, r2, cape) = _eval_metrics(y_test, predictions)

    # Printmetrics
    logger = logging.getLogger(__name__)
    logger.info("Test CAPE: {:.2f}".format(cape))
    logger.info("Test MAE: {:.2f}".format(mae))
    logger.info("Test R2: {:.2f}".format(r2))
    logger.info("Test rmse: {:.2f}".format(rmse))

    ### MLFlow logging ####

    # pre-processing
    mlflow.log_param("selected_features", selected_feat)
    mlflow.log_param("k_best_features", best_svm.get_params().get("univariate_sel__k"))

    # grid search parameters
    if transform_target:
        mlflow.log_param("kernel", best_svm.get_params().get("svm__regressor__kernel"))
        mlflow.log_param("C", best_svm.get_params().get("svm__regressor__C"))
        mlflow.log_param("gamma", best_svm.get_params().get("svm__regressor__gamma"))
        mlflow.log_param(
            "epsilon", best_svm.get_params().get("svm__regressor__epsilon")
        )

    else:
        mlflow.log_param("kernel", best_svm.get_params().get("svm__kernel"))
        mlflow.log_param("C", best_svm.get_params().get("svm__C"))
        mlflow.log_param("gamma", best_svm.get_params().get("svm__gamma"))
        mlflow.log_param("epsilon", best_svm.get_params().get("svm__epsilon"))

    # metrics
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("cape", cape)

    # artifacts
    mlflow.sklearn.log_model(best_svm, "SVM")

    # save model
    _save_model(output_folder, best_svm, wf, alg, predictions)

    return best_svm, X_train, X_test, predictions


def _build_rf(
    X_train: np.ndarray,
    y_train: pd.Series,
    n_splits: int,
    find_max_kbests: bool,
    max_k_bests: int,
    scoring: Dict,
    refit: str,
    transform_target: bool,
) -> object:
    """Trains, cross-validates and tunes a SVM algorithm.

    Args:
        X_train: features training values.
        y_trai: target traning values.
        n_splits: number of splits for CV.
        k_bests: list with the k bests feature to use in feature selection
        scoring: dictionary with the scores for CV multiscoring.
        refit: the metric used to fit data in CV.
        transform_target: whether to transform or not the target.

    Returns:
        An Grid Search CV scikit learn object.

    """
    ctx = context.load_context("../wind-power-forecasting")

    if find_max_kbests:
        k_bests = list(range(1, max_k_bests + 1))
    else:
        k_bests = [max_k_bests]

    selec_k_best = SelectKBest(mutual_info_regression, k=1)
    max_features = list(range(1, max_k_bests + 1))

    if transform_target:
        rf = TransformedTargetRegressor(
            regressor=RandomForestRegressor(bootstrap=True, random_state=42),
            transformer=PowerTransformer(method="yeo-johnson", standardize=True),
            check_inverse=False,
        )
        pipeline = Pipeline(
            [
                (
                    "powertransformer",
                    PowerTransformer(method="yeo-johnson", standardize=True),
                ),
                ("univariate_sel", selec_k_best),
                ("rf", rf),
            ]
        )
        param_grid = {
            "rf__regressor__n_estimators": ctx.params.get("rf_hypms").get(
                "n_estimators"
            ),
            "rf__regressor__max_features": max_features,
            "rf__regressor__max_depth": ctx.params.get("rf_hypms").get("max_depth"),
            "rf__regressor__min_samples_split": ctx.params.get("rf_hypms").get(
                "min_samples_split"
            ),
            "rf__regressor__min_samples_leaf": ctx.params.get("rf_hypms").get(
                "min_samples_leaf"
            ),
            "univariate_sel__k": k_bests,
        }
    else:
        rf = RandomForestRegressor(bootstrap=True, random_state=42)
        pipeline = Pipeline(
            [
                (
                    "powertransformer",
                    PowerTransformer(method="yeo-johnson", standardize=True),
                ),
                ("univariate_sel", selec_k_best),
                ("rf", rf),
            ]
        )
        param_grid = {
            "rf__n_estimators": ctx.params.get("rf_hypms").get("n_estimators"),
            "rf__max_features": max_features,
            "rf__max_depth": ctx.params.get("rf_hypms").get("max_depth"),
            "rf__min_samples_split": ctx.params.get("rf_hypms").get(
                "min_samples_split"
            ),
            "rf__min_samples_leaf": ctx.params.get("rf_hypms").get("min_samples_leaf"),
            "univariate_sel__k": k_bests,
        }

    # Cross validation and hyper-parameter tunning with grid search
    n_splits = n_splits
    refit = refit
    tscv = TimeSeriesSplit(n_splits)
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=tscv, scoring=scoring, refit=refit, n_jobs=-1
    )

    grid_search.fit(X_train, y_train)
    best_rf = grid_search.best_estimator_
    _log_gcv_scores(grid_search, scoring, "RF")

    return best_rf


def _test_best_rf(
    wf,
    best_rf,
    alg,
    X_train,
    y_train,
    X_test,
    y_test,
    feature_names,
    output_folder,
    transform_target,
):

    # Feature selection with the best k value obtanined in Grid Search
    selec_k_best = SelectKBest(
        mutual_info_regression, k=best_rf.get_params().get("univariate_sel__k")
    )
    selec_k_best.fit(X_train, y_train)
    mask = selec_k_best.get_support()  # list of booleans
    selected_feat = []

    for bool, feature in zip(mask, feature_names):
        if bool:
            selected_feat.append(feature)

    predictions = best_rf.predict(X_test)

    # fix negative values in target predictions
    predictions[predictions < 0] = 0.0

    # get metrics
    (rmse, mae, r2, cape) = _eval_metrics(y_test, predictions)

    # Printmetrics
    logger = logging.getLogger(__name__)
    logger.info("Test CAPE: {:.2f}".format(cape))
    logger.info("Test MAE: {:.2f}".format(mae))
    logger.info("Test R2: {:.2f}".format(r2))
    logger.info("Test rmse: {:.2f}".format(rmse))

    ### MLFlow logging ####

    # pre-processing
    mlflow.log_param("selected_features", selected_feat)
    mlflow.log_param("k_best_features", best_rf.get_params().get("univariate_sel__k"))

    # grid search parameters
    if transform_target:
        mlflow.log_param(
            "n_estimators", best_rf.get_params().get("rf__regressor__n_estimators")
        )
        mlflow.log_param(
            "max_features", best_rf.get_params().get("rf__regressor__max_features")
        )
        mlflow.log_param(
            "max_depth", best_rf.get_params().get("rf__regressor__max_depth")
        )
        mlflow.log_param(
            "min_samples_split",
            best_rf.get_params().get("rf__regressor__min_samples_split"),
        )
        mlflow.log_param(
            "min_samples_leaf",
            best_rf.get_params().get("rf__regressor__min_samples_leaf"),
        )

    else:
        mlflow.log_param("n_estimators", best_rf.get_params().get("rf__n_estimators"))
        mlflow.log_param("max_features", best_rf.get_params().get("rf__max_features"))
        mlflow.log_param("max_depth", best_rf.get_params().get("rf__max_depth"))
        mlflow.log_param(
            "min_samples_split", best_rf.get_params().get("rf__min_samples_split")
        )
        mlflow.log_param(
            "min_samples_leaf", best_rf.get_params().get("rf__min_samples_leaf")
        )

    # metrics
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("cape", cape)

    # artifacts
    mlflow.sklearn.log_model(best_rf, "RF")

    # save model
    _save_model(output_folder, best_rf, wf, alg, predictions)

    return best_rf, X_train, X_test, predictions


def _regression_plots(
    wf: str,
    alg: str,
    model: object,
    X_train: np.ndarray,
    y_train: pd.Series,
    X_test: np.ndarray,
    y_test: pd.Series,
    dest_folder: str,
) -> None:
    """Gets several regression related plots.

    Args:
        wf: the Wind Farm.
        model: the model to explore.
        Train and test data sets.

    Returns:
        None, plots show up and are saved
        in data/08_reporting/figures.

    """

    # Get model from received pipeline object
    pt = model[0]
    kbest_tf = model[1]
    model = model.get_params().get(alg.lower())

    X_train = pt.fit_transform(X_train)
    X_test = pt.transform(X_test)

    X_train = kbest_tf.fit_transform(X_train, y_train)
    X_test = kbest_tf.transform(X_test)

    # Residuals plot
    visualizer = ResidualsPlot(model, title=alg, is_fitted=True)
    visualizer.fit(X_train, y_train)
    visualizer.score(X_test, y_test)
    visualizer.show(outpath=dest_folder + "residual_plots.png", clear_figure=True),
    # visualizer.show(clear_figure=True)

    # Prediction error plot
    visualizer = PredictionError(model, title=alg, is_fitted=True)
    visualizer.fit(X_train, y_train)
    visualizer.score(X_test, y_test)
    visualizer.show(outpath=dest_folder + "prediction_error.png", clear_figure=True)
    # visualizer.show(clear_figure=True)


def _validation_plots(
    wf: str,
    alg: str,
    model: object,
    X_train: np.ndarray,
    y_train: pd.Series,
    n_splits: int,
    scorer: object,
    dest_folder: str,
    transform_target: bool,
) -> None:
    """Gets several plots for model validation.

    Args:
        wf: the Wind Farm.
        alg: the algorithm used to create the model.
        model: the model to explore.
        Train and test data sets.
        scorer, f.i., cape_scorer from create_model node.
        transform_target: whether the target is power-transformed or not.

    Returns:
        None, plots show up and are saved
        in data/08_reporting/figures.

    """

    def positive_rmse(estimator, X, y):
        y_pred = estimator.predict(X)
        rmse = mean_squared_error(y, y_pred, squared=False)
        return np.abs(rmse)

    cv = TimeSeriesSplit(n_splits)
    if transform_target:
        model = model.regressor_

    pt = model[0]
    kbest_tf = model[1]
    model = model.get_params().get(alg.lower())
    X_train = pt.fit_transform(X_train)
    X_train = kbest_tf.fit_transform(X_train, y_train)

    # Validation curves
    if alg == "MARS":
        viz = ValidationCurve(
            model,
            param_name="max_terms",
            param_range=np.arange(1, 500, 30),
            cv=cv,
            scoring=positive_rmse,
            title=alg,
        )
    elif alg == "KNN":
        viz = ValidationCurve(
            model,
            param_name="n_neighbors",
            param_range=np.arange(1, 50, 2),
            cv=cv,
            scoring=positive_rmse,
            title=alg,
        )
    elif alg == "SVM":
        viz = ValidationCurve(
            model,
            param_name="epsilon",
            param_range=np.logspace(-5, 0, 12),
            cv=cv,
            scoring=positive_rmse,
            title=alg,
        )
    elif alg == "RF":
        viz = ValidationCurve(
            model,
            param_name="min_samples_leaf",
            param_range=np.logspace(-4, 0, 20, endpoint=True),
            cv=cv,
            scoring=positive_rmse,
            title=alg,
        )

    viz.fit(X_train, y_train)
    viz.show(outpath=dest_folder + "validation_curves.png", clear_figure=True)
    # viz.show(clear_figure=True)

    # Learning curves
    visualizer = LearningCurve(
        model,
        cv=10,
        scoring=positive_rmse,
        title=alg,
        train_sizes=np.linspace(0.1, 1.0, 10),
    )
    visualizer.fit(X_train, y_train)
    visualizer.show(outpath=dest_folder + "learning_curves.png", clear_figure=True)
    # visualizer.show(clear_figure=True)

    # Cross validation scores.
    visualizer = CVScores(
        model,
        cv=cv,
        scoring=positive_rmse,
        title="Errores en validación cruzada para {}".format(alg),
    )
    visualizer.fit(X_train, y_train)
    visualizer.show(outpath=dest_folder + "cv_scores.png", clear_figure=True)
    # visualizer.show(clear_figure=True)


def _time_series_plots(wf, predictions, dest_folder):
    ctx = context.load_context("../wind-power-forecasting/")

    # Load data and feature names for the selected WF
    if wf == "WF1":
        X_test = ctx.catalog.load("X_test_WF1")
        y_test = ctx.catalog.load("y_test_WF1")
        y_test = y_test[0]
    elif wf == "WF2":
        X_test = ctx.catalog.load("X_test_WF2")
        y_test = ctx.catalog.load("y_test_WF2")
        y_test = y_test[0]
    elif wf == "WF3":
        X_test = ctx.catalog.load("X_test_WF3")
        y_test = ctx.catalog.load("y_test_WF3")
        y_test = y_test[0]
    elif wf == "WF4":
        X_test = ctx.catalog.load("X_test_WF4")
        y_test = ctx.catalog.load("y_test_WF4")
        y_test = y_test[0]
    elif wf == "WF5":
        X_test = ctx.catalog.load("X_test_WF5")
        y_test = ctx.catalog.load("y_test_WF5")
        y_test = y_test[0]
    elif wf == "WF6":
        X_test = ctx.catalog.load("X_test_WF6")
        y_test = ctx.catalog.load("y_test_WF6")
        y_test = y_test[0]

    y_test.rename("real", inplace=True)
    predictions = pd.Series(predictions, name="estimado")

    real_pred = pd.concat(
        [pd.to_datetime(X_test.Time, format="%d/%m/%Y %H:%M"), y_test, predictions],
        axis=1,
    )

    fig = px.line(
        data_frame=real_pred,
        x="Time",
        y=["real", "estimado"],
        labels={"Time": "Fecha", "value": "Producción (MWh)"},
        template="plotly_white",
    )
    fig.write_image(dest_folder + "ts_predictions_plot.png")


def _feature_selection_plots(
    wf: str,
    alg: str,
    model: object,
    X_train: np.ndarray,
    y_train: pd.Series,
    n_splits: int,
    scorer: object,
    dest_folder: str,
) -> None:

    # Recursive feature extraction
    model = model.get_params().get(alg.lower())
    cv = TimeSeriesSplit(n_splits)
    visualizer = RFECV(model, cv=cv, scoring=scorer)
    visualizer.fit(X_train, y_train)
    visualizer.show(outpath=dest_folder + "feature_selection.png")
    visualizer.show(clear_figure=True)


def create_model(
    wf: str,
    alg: str,
    find_max_kbests: bool,
    max_k_bests: List,
    n_splits: int,
    transform_target: bool,
    scoring: Dict,
    refit: str,
) -> object:
    """Trains, validates and tunes the chosen algorithm (MARS, KNN, SVR, RF, ANN)
    using CV and GridSearch.

    Args:
        wf: The Wind Farm to model.
        alg: The algorithm to create the model.
        find_max_kbests: whether to add or not max_k_best in hyperparameter optimization.
        max_k_best: Maximum number of k best features.
        algorithm: The alrgorithm to train.
        n_splits: Number of splits to use in cross validation.
        hyperparams: Dictionary contaning the hyperparameters of the algorithm.
        transform_target: Whether to apply or not a target transformation.
        scoring: Dictionary with the list of scores to use in CV.
        refit: The socre to use when refitting with the best params of CV.

    Returns:
        A sklearn GridSearchCV object with all the relevant parameters for
        the creation of the model.

    """

    # load data
    data = _get_data_by_WF(wf)
    X_train = data[0]
    y_train = data[1]

    # Scorers
    cape_scorer = make_scorer(metric.get_cape, greater_is_better=False)
    scoring["CAPE"] = cape_scorer
    refit = refit

    if alg == "MARS":
        best_model = _build_mars(
            X_train,
            y_train,
            n_splits,
            find_max_kbests,
            max_k_bests,
            scoring,
            refit,
            transform_target,
        )

    elif alg == "KNN":
        best_model = _build_knn(
            X_train,
            y_train,
            n_splits,
            find_max_kbests,
            max_k_bests,
            scoring,
            refit,
            transform_target,
        )
    elif alg == "SVM":
        best_model = _build_svm(
            X_train,
            y_train,
            n_splits,
            find_max_kbests,
            max_k_bests,
            scoring,
            refit,
            transform_target,
        )
    elif alg == "RF":
        best_model = _build_rf(
            X_train,
            y_train,
            n_splits,
            find_max_kbests,
            max_k_bests,
            scoring,
            refit,
            transform_target,
        )

    return wf, alg, best_model, X_train, y_train


def test_model(
    alg: str,
    wf: str,
    best_model: object,
    X_train: np.ndarray,
    y_train: pd.Series,
    output_folder: str,
    transform_target: bool,
) -> object:
    """Re-trains the model on full X_train data set used in GridSearchCV
    with the best hyperparameters and tests it on X_test.

        Args:
            gcv: GridSearchObject with the results of the CV + tunning.
            X_train: The training set
            y_train: The target traning values.

        Returns:
            The model re-trained ready to predict on new unseen data.
            Train and test data sets.
            The Wind Farm identification.

    """

    # Load test data
    data = _get_data_by_WF(wf)
    X_test = data[2]
    y_test = data[3]
    feature_names = data[4]

    if alg == "MARS":
        model, X_train, X_test, predictions = _test_best_mars(
            wf,
            best_model,
            alg,
            X_train,
            y_train,
            X_test,
            y_test,
            feature_names,
            output_folder,
            transform_target,
        )
    elif alg == "KNN":
        model, X_train_, X_test, predictions = _test_best_knn(
            wf,
            best_model,
            alg,
            X_train,
            y_train,
            X_test,
            y_test,
            feature_names,
            output_folder,
            transform_target,
        )
    elif alg == "SVM":
        model, X_train, X_test, predictions = _test_best_svm(
            wf,
            best_model,
            alg,
            X_train,
            y_train,
            X_test,
            y_test,
            feature_names,
            output_folder,
            transform_target,
        )
    elif alg == "RF":
        model, X_train, X_test, predictions = _test_best_rf(
            wf,
            best_model,
            alg,
            X_train,
            y_train,
            X_test,
            y_test,
            feature_names,
            output_folder,
            transform_target,
        )

    return model, X_train, X_test, predictions


def get_model_plots(
    wf: str,
    alg: str,
    model: object,
    predictions: np.ndarray,
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_splits: int,
    scorer: object,
    folder: str,
) -> None:
    """Several plots related to model analysis.
    - Regression related plots
    - Learning and validation plots.
    - Feature extraction plots.
    """
    ctx = context.load_context("../wind-power-forecasting")
    transform_target = ctx.params.get("transform_target")

    data = _get_data_by_WF(wf)
    y_train = data[1]
    y_test = data[3]

    # Create destination folder
    os.makedirs(folder + "figures/" + wf + "/" + alg + "/", exist_ok=True)
    dest_folder = folder + "figures/" + wf + "/" + alg + "/"

    _regression_plots(
        wf, alg, model, X_train, y_train, X_test, y_test, dest_folder,
    )
    _validation_plots(
        wf,
        alg,
        model,
        X_train,
        y_train,
        n_splits,
        scorer,
        dest_folder,
        transform_target,
    )

    _time_series_plots(wf, predictions, dest_folder)

    # Log artifacts with MLflow
    mlflow.log_artifacts(dest_folder)
