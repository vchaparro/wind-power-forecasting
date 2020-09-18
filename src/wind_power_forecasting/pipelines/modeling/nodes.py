import pandas as pd
import pickle
from kedro.framework import context
from functools import wraps
from typing import Callable, Dict, List
import time
import logging
import numpy as np
from pathlib import Path
import datetime as dt
import os
import re
import matplotlib.pyplot as plt
from typing import List, Dict
from wind_power_forecasting.nodes import metric
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
from mlflow import log_metric, log_param, log_artifact
from sklearn.preprocessing import PowerTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.metrics.scorer import make_scorer
from pyearth import Earth
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import mlflow
import mlflow.sklearn
from mlflow import log_metric, log_param, log_artifact
from yellowbrick.regressor import ResidualsPlot, PredictionError
from yellowbrick.model_selection import ValidationCurve, LearningCurve, CVScores, RFECV


def _log_gcv_scores(gcv: object, scoring: Dict) -> Dict:
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
    logger.info("Best hyperparameters found for MARS: {}".format(gcv.best_params_))


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


def _save_model(folder: str, model, WF: str) -> None:
    """ Saves the trained model.

        Args: 
            folder: the folder where the pickle objects will be saved.
            WF: Wind Farm identification.
            model: the model object.
            
        Returns:
            None.
    """
    os.makedirs(folder + WF, exist_ok=True)

    with open(folder + "{}/mars.pickle".format(WF), "wb") as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _build_mars(
    X_train: np.ndarray,
    y_train: pd.Series,
    n_splits: int,
    k_bests: List,
    scoring: Dict,
    refit: str,
    hyperparams: Dict,
) -> object:
    """ Trains, cross-validates and tunes a 
        Multivariate Adaptative Regression Splines (MARS) algorithm.

        Args:
            X_train: features training values.
            y_trai: target traning values.
            n_splits: number of splits for CV.
            k_bests: list with the k bests feature to use in feature selection
            scoring: dictionary with the scores for CV multiscoring.
            refit: the metric used to fit data in CV.
            hyperparams: mars hyperparameters to be tuned in grid search CV.

        Returns:
            An Grid Search CV scikit learn object.
            
    """
    selec_k_best = SelectKBest(mutual_info_regression, k=1)
    pipeline = Pipeline(
        [
            ("univariate_sel", selec_k_best),
            ("mars", Earth(feature_importance_type="gcv")),
        ]
    )
    param_grid = {
        "mars__max_degree": hyperparams.get("max_degree"),
        "mars__allow_linear": hyperparams.get("allow_linear"),
        "mars__penalty": hyperparams.get("penalty"),
        "univariate_sel__k": k_bests,
    }

    # Cross validation and hyper-parameter tunning with grid search
    n_splits = n_splits
    refit = refit
    tscv = TimeSeriesSplit(n_splits)
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=tscv, scoring=scoring, refit=refit, n_jobs=-1
    )

    return grid_search.fit(X_train, y_train)


def _train_test_best_mars(
    wf, gcv, X_train, y_train, X_test, y_test, feature_names, output_folder,
):

    # Feature selection with the best k value obtanined in Grid Search
    selec_k_best = SelectKBest(
        mutual_info_regression, k=gcv.best_params_["univariate_sel__k"]
    )
    selec_k_best.fit(X_train, y_train)
    X_train_2 = selec_k_best.transform(X_train)
    X_test_2 = selec_k_best.transform(X_test)

    mask = selec_k_best.get_support()  # list of booleans
    selected_feat = []

    for bool, feature in zip(mask, feature_names):
        if bool:
            selected_feat.append(feature)

    # Re-training without CV, using the best parameters obtained by CV
    mars = Earth(
        max_degree=gcv.best_params_["mars__max_degree"],
        allow_linear=gcv.best_params_["mars__allow_linear"],
        penalty=gcv.best_params_["mars__penalty"],
    )

    mars.fit(X_train_2, y_train)
    predictions = mars.predict(X_test_2)

    # build prediction matrix (ID,Production)
    pred_matrix = np.stack(
        (np.array(y_test.index.to_series()).astype(int), predictions), axis=-1
    )
    df_pred = pd.DataFrame(
        data=pred_matrix.reshape(-1, 2), columns=["ID", "Production"]
    )

    # get metrics
    (rmse, mae, r2, cape) = _eval_metrics(y_test, predictions)

    # Printmetrics
    logger = logging.getLogger(__name__)
    logger.info("Test CAPE: {:.2f}".format(cape))
    logger.info("Test MAE: {:.2f}".format(mae))
    logger.info("Test R2: {:.2f}".format(r2))
    logger.info("Test rmse: {:.2f}".format(rmse))

    ### MLFlow logging ####
    mlflow.set_tag("grid_searh_best_params", gcv.best_params_)

    # pre-processing
    mlflow.log_param("selected_features", selected_feat)
    mlflow.log_param("k_best_features", gcv.param_grid.get("univariate_sel__k"))

    # grid search parameters
    mlflow.log_param("max_degree", gcv.param_grid.get("mars__max_degree"))
    mlflow.log_param("penalty", gcv.param_grid.get("mars__penalty"))
    mlflow.log_param("allow_linear", gcv.param_grid.get("mars__allow_linear"))
    mlflow.log_param("n_splits", gcv.n_splits_)

    # metrics
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("cape", cape)

    # artifacts
    mlflow.sklearn.log_model(mars, "MARS")
    # mlflow.log_artifacts()

    # save model
    _save_model(output_folder, mars, wf)

    return mars, X_train_2, X_test_2


def _regression_plots(
    wf: str,
    algorithm: str,
    model: object,
    X_train: np.ndarray,
    y_train: pd.Series,
    X_test: np.ndarray,
    y_test: pd.Series,
    dest_folder: str,
) -> None:
    """ Gets several regression related plots.

        Args:
            wf: the Wind Farm.
            algorithm: the algorithm used to craete the model.
            model: the model to explore.
            Train and test data sets.

        Returns:
            None, plots show up and are saved 
            in data/08_reporting/figures.
            
    """

    # Residuals plot
    visualizer = ResidualsPlot(model)
    visualizer.fit(X_train, y_train)
    visualizer.score(X_test, y_test)
    visualizer.show(outpath=dest_folder + "residual_plots.png"),
    visualizer.show(clear_figure=True)

    # Prediction error plot
    visualizer = PredictionError(model)
    visualizer.fit(X_train, y_train)
    visualizer.score(X_test, y_test)
    visualizer.show(outpath=dest_folder + "prediction_error.png")
    visualizer.show(clear_figure=True)


def _validation_plots(
    wf: str,
    algorithm: str,
    model: object,
    X_train: np.ndarray,
    y_train: pd.Series,
    n_splits: int,
    scorer: object,
    dest_folder: str,
) -> None:
    """ Gets several plots for model validation.

        Args:
            wf: the Wind Farm.
            algorithm: the algorithm used to create the model.
            model: the model to explore.
            Train and test data sets.
            scorer, f.i., cape_scorer from create_model node.

        Returns:
            None, plots show up and are saved 
            in data/08_reporting/figures.
            
    """
    cv = TimeSeriesSplit(n_splits)
    if algorithm == "MARS":

        # Validation curves
        viz = ValidationCurve(
            model,
            param_name="max_degree",
            param_range=np.arange(1, 11),
            cv=cv,
            scoring=scorer,
        )

        viz.fit(X_train, y_train)
        viz.show(outpath=dest_folder + "validation_curves.png")
        viz.show(clear_figure=True)

        # Learning curves
        visualizer = LearningCurve(
            model, scoring=scorer, train_sizes=np.linspace(0.1, 1.0, 7)
        )
        visualizer.fit(X_train, y_train)
        visualizer.show(outpath=dest_folder + "learning_curves.png")
        visualizer.show(clear_figure=True)

        # Cross validation scores.
        visualizer = CVScores(model, cv=cv, scoring=scorer,)
        visualizer.fit(X_train, y_train)
        visualizer.show(outpath=dest_folder + "cv_scores.png")
        visualizer.show(clear_figure=True)


def _feature_selection_plots(
    wf: str,
    algorithm: str,
    model: object,
    X_train: np.ndarray,
    y_train: pd.Series,
    n_splits: int,
    scorer: object,
    dest_folder: str,
) -> None:

    # Recursive feature extraction
    cv = TimeSeriesSplit(n_splits)
    visualizer = RFECV(model, cv=cv, scoring=scorer)
    visualizer.fit(X_train, y_train)
    visualizer.show(outpath=dest_folder + "feature_selection.png")
    visualizer.show(clear_figure=True)


def create_model(
    wf: str, algorithm: str, k_bests: List, n_splits: int, hyperparams: Dict,
) -> object:
    """ Trains, validates and tunes the chosen algorithm (MARS, KNN, SVR, RF, ANN)
        using CV and GridSearch.

        Args:
            wf: The Wind Farm to model.
            k_best: Number of k best features.
            algorithm: The alrgorithm to train.
            n_splits: Number of splits to use in cross validation.
            hyperparams: Dictionary contaning the hyperparameters of the algorithm.

        Returns:
            A sklearn GridSearchCV object with all the relevant parameters for
            the creation of the model.
            
    """
    # load data
    data = _get_data_by_WF(wf)
    X_train = data[0]
    y_train = data[1]
    X_test = data[2]
    y_test = data[3]
    feature_names = data[4]

    # Scorers
    cape_scorer = make_scorer(metric.get_cape, greater_is_better=False)
    scoring = {
        "RMSE": "neg_root_mean_squared_error",
        "MAE": "neg_mean_absolute_error",
        "R2": "r2",
        "CAPE": cape_scorer,
    }
    refit = "CAPE"

    # Init gcv object to a default value
    gcv = GridSearchCV(Earth(), param_grid={"max_degree": [2]})
    if algorithm == "MARS":

        gcv = _build_mars(
            X_train, y_train, n_splits, k_bests, scoring, refit, hyperparams
        )

        # Log CV score results.
        _log_gcv_scores(gcv, scoring)

    return wf, algorithm, gcv, X_train, y_train, X_test, y_test, cape_scorer


def train_test_model(
    algorithm: str,
    wf: str,
    gcv: object,
    X_train: np.ndarray,
    y_train: pd.Series,
    output_folder: str,
) -> object:
    """ Re-trains the model on full X_train data set used in GridSearchCV
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

    if algorithm == "MARS":
        model, X_train_pped, X_test_pped = _train_test_best_mars(
            wf, gcv, X_train, y_train, X_test, y_test, feature_names, output_folder,
        )

    return model, X_train_pped, X_test_pped


def get_model_plots(
    wf: str,
    algorithm: str,
    model: object,
    X_train: np.ndarray,
    y_train: pd.Series,
    X_test: np.ndarray,
    y_test: pd.Series,
    X_train_2: np.ndarray,
    X_test_2: np.ndarray,
    n_splits: int,
    scorer: object,
    folder: str,
) -> None:
    """ Several plots related to model analysis.
        - Regression related plots
        - Learning and validation plots.
        - Feature extraction plots.
    """

    # Create destination folder
    os.makedirs(folder + "figures/" + wf + "/" + algorithm + "/", exist_ok=True)
    dest_folder = folder + "figures/" + wf + "/" + algorithm + "/"

    _regression_plots(
        wf, algorithm, model, X_train_2, y_train, X_test_2, y_test, dest_folder,
    )
    _validation_plots(
        wf, algorithm, model, X_train_2, y_train, n_splits, scorer, dest_folder,
    )

    # Only applies for stimators with feature_importance parameter
    # _feature_selection_plots(
    #    wf, algorithm, model, X_train, y_train, n_splits, scorer, dest_folder,
    # )

