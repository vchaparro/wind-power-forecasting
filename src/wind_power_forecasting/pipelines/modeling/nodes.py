import pandas as pd
import pickle
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
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.metrics.scorer import make_scorer
from pyearth import Earth
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def create_model(
    wf: str,
    algorithm: str,
    k_bests: List,
    n_splits: int,
    X_train: np.ndarray,
    y_train: pd.DataFrame,
    feat_names: List,
    hyperparams: Dict,
):
    """ Trains, validates and tunes the chosen algorithm (MARS, KNN, SVR, RF, ANN)
        using CV and GridSearch.

        Args:
            wf: The Wind Farm to model.
            k_best: Number of k best features.
            algorithm: The alrgorithm to train.
            n_splits: Number of splits to use in cross validation.
            X_train: Traning data for the selected WF.
            y_train: Target data for the selected WF.
            feat_names: A list containing the names of the features resulted from
                feauture engineering step.

        Returns:
            A sklearn GridSearch object with all the relevant parameters for
            the creation of the model.
            
    """
    y_train = y_train[0]

    # Feature selection
    feature_names = feat_names
    selec_k_best = SelectKBest(mutual_info_regression, k=1)

    # Scorers
    cape_scorer = make_scorer(metric.get_cape, greater_is_better=False)
    scoring = {
        "RMSE": "neg_root_mean_squared_error",
        "MAE": "neg_mean_absolute_error",
        "R2": "r2",
        "CAPE": cape_scorer,
    }

    if algorithm == "MARS":
        # Get algorithm hiperparameters
        pipeline = Pipeline(
            [
                ("univariate_sel", selec_k_best),
                ("mars", Earth(feature_importance_type="gcv")),
            ]
        )
        ## Modeling: MARS using py-earth ######

        param_grid = {
            "mars__max_degree": hyperparams.get("max_degree"),
            "mars__allow_linear": hyperparams.get("allow_linear"),
            "mars__penalty": hyperparams.get("penalty"),
            "univariate_sel__k": k_bests,
        }

        # Cross validation and hyper-parameter tunning with grid search
        n_splits = n_splits
        tscv = TimeSeriesSplit(n_splits)
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=tscv, scoring=scoring, refit="CAPE", n_jobs=-1
        )

        gcv = grid_search.fit(X_train, y_train)

        # Log CV score results.
        results = gcv.cv_results_
        logger = logging.getLogger(__name__)

        for scorer in sorted(scoring):
            best_index = np.nonzero(results["rank_test_%s" % scorer] == 1)[0][0]
            best_CV = results["mean_test_%s" % scorer][best_index]
            best_CV_std = results["std_test_%s" % scorer][best_index]

            if (scorer == "RMSE") or (scorer == "MAE") or (scorer == "CAPE"):
                logger.info(
                    "Best CV {0}:{1:.2f}, std:{2:.2f}".format(
                        scorer, -best_CV, best_CV_std
                    )
                )
            else:
                logger.info(
                    "Best CV {0}:{1:.2f}, std:{2:.2f}".format(
                        scorer, best_CV, best_CV_std
                    )
                )

        # Log best hyperparameters found
        logger.info("Best hyperparameters found for MARS: {}".format(gcv.best_params_))

    return gcv

