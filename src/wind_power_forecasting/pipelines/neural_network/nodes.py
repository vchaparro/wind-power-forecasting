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
from wind_power_forecasting.nodes import data_transformation as dtr
from wind_power_forecasting.pipelines.feature_engineering.nodes import (
    feature_engineering as fe,
)
from wind_power_forecasting.pipelines.modeling.nodes import (
    _get_data_by_WF,
    _log_gcv_scores,
    _eval_metrics,
    _save_model,
)
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
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from yellowbrick.regressor import ResidualsPlot, PredictionError
from yellowbrick.model_selection import ValidationCurve, LearningCurve, CVScores, RFECV
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow import keras

import plotly
import cufflinks as cf

def _regression_plots(
    wf: str,
    alg: str,
    model_folder: str,
    X_train: np.ndarray,
    y_train: pd.Series,
    X_test: np.ndarray,
    y_test: pd.Series,
    dest_folder: str,
) -> None:
    """ Gets several regression related plots.

        Args:
            wf: the Wind Farm.
            model: the model to explore.
            Train and test data sets.

        Returns:
            None, plots show up and are saved 
            in data/08_reporting/figures.
            
    """
    # Get model from received pipeline object
    model = tf.keras.load_model(model_folder +  + "{0}/{1}.h5".format(wf, "ANN"))

    # Residuals plot
    visualizer = ResidualsPlot(model, title=alg)
    visualizer.fit(X_train, y_train)
    visualizer.score(X_test, y_test)
    visualizer.show(outpath=dest_folder + "residual_plots.png"),
    visualizer.show(clear_figure=True)

    # Prediction error plot
    visualizer = PredictionError(model, title=alg)
    visualizer.fit(X_train, y_train)
    visualizer.score(X_test, y_test)
    visualizer.show(outpath=dest_folder + "prediction_error.png")
    visualizer.show(clear_figure=True)


def _validation_plots(
    wf: str,
    alg: str,
    model_folder: object,
    X_train: np.ndarray,
    y_train: pd.Series,
    n_splits: int,
    scorer: object,
    dest_folder: str,
    transform_target: bool,
) -> None:
    """ Gets several plots for model validation.

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
    model = tf.keras.load_model(model_folder +  + "{0}/{1}.h5".format(wf, "ANN"))
    cv = TimeSeriesSplit(n_splits)
    if transform_target:
        model = model.regressor_

    

    # Learning curves
    visualizer = LearningCurve(
        model, scoring=scorer, train_sizes=np.linspace(0.1, 1.0, 20), title=alg,
    )
    visualizer.fit(X_train, y_train)
    visualizer.show(outpath=dest_folder + "learning_curves.png")
    visualizer.show(clear_figure=True)

    # Cross validation scores.
    visualizer = CVScores(
        model,
        cv=cv,
        scoring=scorer,
        title="Errores en validación cruzada para {}".format(alg),
    )
    visualizer.fit(X_train, y_train)
    visualizer.show(outpath=dest_folder + "cv_scores.png")
    visualizer.show(clear_figure=True)


def _build_ann(n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=[3]):
    import tensorflow as tf

    ann = tf.keras.models.Sequential()
    ann.add(tf.keras.layers.InputLayer(input_shape=input_shape))

    for layer in range(n_hidden):
        ann.add(tf.keras.layers.Dense(n_neurons, activation="relu"))

    ann.add(tf.keras.layers.Dense(1))
    optimizer = tf.keras.optimizers.SGD(lr=learning_rate)
    ann.compile(loss="mse", optimizer=optimizer, metric="r2")

    return ann


def _get_run_logdir():
    import time

    root_logdir = os.path.join(os.curdir, "ann_logs")
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")

    return os.path.join(root_logdir, run_id)


def train_test_ann(
    wf: str,
    n_splits: int,
    find_max_kbests: bool,
    max_k_bests: int,
    transform_target: bool,
    scoring: Dict,
    refit: str,
) -> object:
    """ Trains, cross-validates and tunes an ANN algorithm.

        Args:
            wf: Wind Farm identificator.
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
    ctx = context.load_context("../wind-power-forecasting")
    data = _get_data_by_WF(wf)
    X_train = data[0]
    y_train = data[1]
    X_test = data[2]
    y_test = data[3]
    feature_names = data[4]
    output_folder = ctx.params.get("folder").get("mdl")

    if find_max_kbests:
        k_bests = list(range(1, max_k_bests + 1))
    else:
        k_bests = [max_k_bests]

    selec_k_best = SelectKBest(mutual_info_regression, k=1)

    if transform_target:
        nn = keras.wrappers.scikit_learn.KerasRegressor(
            _build_ann, epochs=10, batch_size=3
        )
        ann = TransformedTargetRegressor(
            regressor=nn,
            transformer=PowerTransformer(method="yeo-johnson", standardize=True),
            check_inverse=False,
        )

        pipeline = Pipeline([("univariate_sel", selec_k_best), ("ann", ann)])
        param_grid = {
            "ann__regressor__n_hidden": ctx.params.get("ann_hypms").get("n_hidden"),
            "ann__regressor__n_neurons": ctx.params.get("ann_hypms").get("n_neurons"),
            "ann__regressor__learning_rate": ctx.params.get("ann_hypms").get(
                "learning_rate"
            ),
            "ann_regressor_input_shape": [max_k_bests],
            "univariate_sel__k": k_bests,
        }
    else:
        ann = keras.wrappers.scikit_learn.KerasRegressor(
            _build_ann, epochs=10, batch_size=3
        )
        pipeline = Pipeline([("univariate_sel", selec_k_best), ("ann", ann)])
        param_grid = {
            "ann__n_hidden": ctx.params.get("ann_hypms").get("n_hidden"),
            "ann__n_neurons": ctx.params.get("ann_hypms").get("n_neurons"),
            "ann__learning_rate": ctx.params.get("ann_hypms").get("learning_rate"),
            "ann__input_shape": [max_k_bests],
            "univariate_sel__k": k_bests,
        }

    # Scorers
    cape_scorer = make_scorer(metric.get_cape, greater_is_better=False)
    scoring["CAPE"] = cape_scorer
    refit = refit

    # Cross validation and hyper-parameter tunning with grid search
    n_splits = n_splits
    tscv = TimeSeriesSplit(n_splits)

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=tscv,
        n_jobs=-1,
        verbose=100,
        scoring=scoring,
        refit=refit,
    )

    grid_search.fit(X_train, y_train.to_numpy())
    best_ann = grid_search.best_estimator_
    _log_gcv_scores(grid_search, scoring, "ANN")

    # Feature selection with the best k value obtanined in Grid Search
    selec_k_best = SelectKBest(
        mutual_info_regression, k=best_ann.get_params().get("univariate_sel__k")
    )
    selec_k_best.fit(X_train, y_train)
    X_train_2 = selec_k_best.transform(X_train)
    X_test_2 = selec_k_best.transform(X_test)

    mask = selec_k_best.get_support()  # list of booleans
    selected_feat = []

    for bool, feature in zip(mask, feature_names):
        if bool:
            selected_feat.append(feature)

    # Plot learning curves when refitting
    run_logdir = _get_run_logdir()
    tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)
    history = best_ann.named_steps.get("ann").model.fit(
        X_train_2, y_train.to_numpy(), epochs=30, callbacks=[tensorboard_cb]
    )

    predictions = best_ann.predict(X_test)

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

    # pre-processing
    mlflow.log_param("selected_features", selected_feat)
    mlflow.log_param("k_best_features", best_ann.get_params().get("univariate_sel__k"))

    # grid search parameters
    if transform_target:
        mlflow.log_param(
            "n_hidden", best_ann.get_params().get("ann__regressor__n_hidden")
        )
        mlflow.log_param(
            "n_neurons", best_ann.get_params().get("ann__regressor__n_neurons")
        )
        mlflow.log_param(
            "learning_rate", best_ann.get_params().get("ann__regressor__learning_rate")
        )
    else:
        mlflow.log_param("n_hidden", best_ann.get_params().get("ann__n_hidden"))
        mlflow.log_param("n_neurons", best_ann.get_params().get("ann__n_neurons"))
        mlflow.log_param(
            "learning_rate", best_ann.get_params().get("ann__learning_rate")
        )

    # metrics
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("cape", cape)

    # artifacts
    # mlflow.sklearn.log_model(best_ann, "ANN")

    # save model
    best_ann.named_steps.get("ann").model.save(
        output_folder + "{0}/{1}.h5".format(wf, "ANN")
    )

    return predictions

def get_nn_plots(
    wf: str,
    alg: str,
    model: object,
    predictions: np.ndarray,
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
    ctx = context.load_context("../wind-power-forecasting")
    transform_target = ctx.params.get("transform_target")

    data = _get_data_by_WF(wf)
    y_train = data[1]
    y_test = data[3]

    # Create destination folder
    os.makedirs(folder + "figures/" + wf + "/" + alg + "/", exist_ok=True)
    dest_folder = folder + "figures/" + wf + "/" + alg + "/"

    _regression_plots(
        wf, alg, model, X_train_2, y_train, X_test_2, y_test, dest_folder,
    )
    _validation_plots(
        wf,
        alg,
        model,
        X_train_2,
        y_train,
        n_splits,
        scorer,
        dest_folder,
        transform_target,
    )

    _time_series_plots(wf, predictions)

