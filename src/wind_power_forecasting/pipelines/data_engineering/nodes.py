import datetime as dt
import logging
import os
import re
import time
from functools import wraps
from pathlib import Path
from typing import Callable, Dict, List

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from kedro.framework import context
from metpy import calc
from metpy.units import units
from operational_analysis.toolkits import filters, power_curve


def log_running_time(func: Callable) -> Callable:
    """Decorator for logging node execution time.
    Args:
        func: Function to be executed.

    Returns:
        Decorator for logging the running time.
    """

    @wraps(func)
    def with_time(*args, **kwargs):
        log = logging.getLogger(__name__)
        t_start = time.time()
        result = func(*args, **kwargs)
        t_end = time.time()
        elapsed = t_end - t_start
        log.info("Running %r took %.2f seconds", func.__name__, elapsed)
        return result

    return with_time


def _get_wind_speed(x: pd.DataFrame) -> float:
    """Function to get wind speed from wind velocity components U and V.

    Args:
        x: Feature data frame containing components U and V as columns.

    Returns:
        Wind speed for each pair <U,V>.

    """

    return float(
        calc.wind_speed(
            x.U * units.meter / units.second, x.V * units.meter / units.second
        ).magnitude
    )


def _save_fig(
    fig_id: int, folder: str, tight_layout=True, fig_extension="png", resolution=300,
):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, fig_id + "." + fig_extension)

    if tight_layout:
        plt.tight_layout()

    plt.savefig(path, format=fig_extension, dpi=resolution)
    mlflow.log_artifacts(folder)


def _plot_flagged_pc(ws, p, flag_bool, alpha):
    plt.scatter(ws, p, s=3, alpha=alpha)
    plt.scatter(ws[flag_bool], p[flag_bool], s=3, c="red")
    plt.xlabel("velocidad (m/s)", fontsize=20)
    plt.ylabel("potencia (MWh)", fontsize=20)


def get_data_by_wf(X: pd.DataFrame, y: pd.Series, wf: str) -> pd.DataFrame:
    """Get data filterd by Wind Farm (wf).

    Args:
        X: X_train_raw.
        **y: y_train_raw (optional)
        wf: Wind Farm identification.

    Returns:
        X, y data frames filtered by Wind Farm.
    """

    # Row selection by WF
    X = X[X["WF"] == wf]
    X["Time"] = pd.to_datetime(X["Time"], format="%d/%m/%Y %H:%M")

    X = pd.merge(X, y, on="ID", how="inner")
    y = X["Production"]
    del X["Production"]

    return X, y


def split_data_by_date(date: str, X: pd.DataFrame, y: pd.Series) -> Dict:
    """It splits X and y sets by a 'Time' value  into sets for training and testing.

    Args:
        X: cleaned X_train features data frame.
        y: cleaned y_train target dta frame.

    Returns:
        A dictionary with the four sets (X_train, y_train, X_test, y_test).
    """
    sets = {}
    date_cut = dt.datetime.strptime(date, "%Y-%m-%d %H:%M:%S")

    X_test = X[X["Time"] > date_cut]
    X_train = X[X["Time"] <= date_cut]
    y_train = y[X_train.index]
    y_test = y[X_test.index]

    sets["X_train"] = X_train
    sets["X_test"] = X_test
    sets["y_train"] = y_train
    sets["y_test"] = y_test

    sets["y_train"].name = None
    sets["y_test"].name = None

    return sets


def add_new_cols(new_cols: list, data_sets: Dict) -> pd.DataFrame:
    """Adds new columns to a given data frame.

    Args:
        new_cols: list with the column names to be added.
        X_train: train data set to be expanded with new_cols.
        X_train: test data set to be expanded with new_cols.

    Returns:
        X_train and X_test expanded with the new columns and the columns that
        contains missing values.

    """
    X_train = data_sets.get("X_train")
    X_test = data_sets.get("X_test")

    # For predictions only can be used data avialable on day D at 09h. --> columns 3 to -9
    X_test = X_test[X_test.columns[0:-9]]

    for col in new_cols:
        X_train[col] = np.nan
        X_test[col] = np.nan

    return X_train, X_test


def _interpolate_missing_values(X: pd.DataFrame, cols: List) -> pd.DataFrame:
    """Imputes those missing values due to the NWP's frequency in data providing.

    Args:
        X: Data frame where missing values are being inputed.
        cols: list of column names of the data frame that have missing values.

    Returns:
        None, missing values are inputed inplace.

    """
    X.index = X["Time"]
    del X["Time"]

    for var in cols:
        X[var].interpolate(
            method="time", inplace=True, limit=100, limit_direction="both"
        )

    X.reset_index(inplace=True)

    return X


@log_running_time
def _daily_missing_values(X: pd.DataFrame, cols: List) -> pd.DataFrame:
    """Impute missing values based on the gap time between forecasted timestamp and NWP run.

    Args:
        X: the data frame where the missing will be inputed.
        cols: columns with missig values due to daily frequency of NWP.

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

    for col in reversed(cols):
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
                    X["NWP" + key + "_" + i] = X["NWP" + key + "_" + i].fillna(
                        X[col_name]
                    )

    return X


def input_missing_values(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
    """ Inputs the missing values in both training and test sets.

            Args:
                X_train/X_test data.
                
            
            Returns:
                X_train/test with filled in missing values.
    """

    ctx = context.load_context("../wind-power-forecasting")
    cols_to_interpol = ctx.params.get("cols_to_interpol")
    X_train_raw = ctx.catalog.load("X_train_raw")
    cols_train = X_train_raw.columns[3:]
    cols_test = X_train_raw.columns[3:-9]

    X_train_no_missings = _daily_missing_values(X_train, cols_train)
    X_test_no_missings = _daily_missing_values(X_test, cols_test)
    X_train_no_missings = _interpolate_missing_values(
        X_train_no_missings, cols_to_interpol
    )
    X_test_no_missings = _interpolate_missing_values(X_test, cols_to_interpol)

    return X_train_no_missings, X_test_no_missings


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
    X_train["U"] = (X_train.NWP1_U + X_train.NWP2_U + X_train.NWP3_U) / 3
    X_train["V"] = (X_train.NWP1_V + X_train.NWP2_V + X_train.NWP3_V) / 3
    X_train["T"] = (X_train.NWP1_T + X_train.NWP3_T) / 2
    X_train["CLCT"] = X_train.NWP4_CLCT

    X_test["U"] = (X_test.NWP1_U + X_test.NWP2_U + X_test.NWP3_U) / 3
    X_test["V"] = (X_test.NWP1_V + X_test.NWP2_V + X_test.NWP3_V) / 3
    X_test["T"] = (X_test.NWP1_T + X_test.NWP3_T) / 2
    X_test["CLCT"] = X_test.NWP4_CLCT

    # Select final features
    X_train = X_train[["ID", "Time", "U", "V", "T", "CLCT"]]
    X_test = X_test[["ID", "Time", "U", "V", "T", "CLCT"]]

    return X_train, X_test


def _find_outliers(X: pd.DataFrame, y: pd.Series, wf: str, *args) -> Dict[str, list]:
    """Finds outliers based on power curve, using a binning method.

    Args:
        X: Feature data frame.
        y: target.
        parameters: dictionary containing the configuration parameters.

    Returns:
        outliers: dictionary with the different type of outliers found.

    """
    # Loading context
    ctx = context.load_context("../wind-power-forecasting/")

    # Dictionary to save boolean outlier marker
    outliers = {}

    # Power curve data
    X["Production"] = y.to_list()
    X["wspeed"] = X.apply(_get_wind_speed, axis=1)
    X_ = X[["wspeed", "Production"]]

    # Select appropiate values for binning method parameters.
    top_frac_max = ctx.params.get(wf).get("top_frac_max")
    sparse_bin_width = ctx.params.get(wf).get("sparse_bin_width")
    frac_std = ctx.params.get(wf).get("frac_std")
    threshold_type = ctx.params.get(wf).get("threshold_type")
    bottom_max = ctx.params.get(wf).get("bottom_max")

    # top-curve stacked outliers
    top = filters.window_range_flag(
        X_.Production,
        top_frac_max * X_.Production.max(),
        X_.Production.max(),
        X_.wspeed,
        12.5,
        2000.0,
    )

    # sparse outliers
    max_bin = 0.99 * X_["Production"].max()
    sparse = filters.bin_filter(
        X_.Production,
        X_.wspeed,
        sparse_bin_width,
        frac_std * X_.Production.std(),
        "median",
        0.025,
        max_bin,
        threshold_type,
        "all",
    )

    # bottom-curve stacked outliers
    bottom = filters.window_range_flag(
        X_.wspeed, bottom_max, 40, X_.Production, 0.025, 2000.0
    )

    # Plot outliers
    _plot_flagged_pc(
        X_.wspeed, X_.Production, (top) | (sparse) | (bottom), 0.3,
    )

    if args:
        fig_id = args[0]
    else:
        fig_id = "outliers"

    _save_fig(
        fig_id, ctx.params.get("folder").get("rep") + "figures/" + wf + "/",
    )
    plt.show()
    plt.close()

    # Populate the dictionary
    outliers["top"] = top
    outliers["sparse"] = sparse
    outliers["bottom"] = bottom

    del X["Production"], X["wspeed"], X_

    return outliers


@log_running_time
def clean_outliers(X: pd.DataFrame, sets: Dict, wf: str, *args) -> tuple:
    """It removes the outliers and returned cleaned X, y.

    Args:
        X: the feature data frame.
        y: the target.

    Returns:
        Cleaned X, y.

    """
    y = sets.get("y_train")

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


def fix_negative_values(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
    """Replaces negative values of CLCT by 0.

    Args:
        X_train/test: the data frame containing CLCT column.
    Returns:
        The data sets with CLCT negative values fixed.

    """
    X_train.loc[X_train["CLCT"] < 0, "CLCT"] = 0.0
    X_test.loc[X_test["CLCT"] < 0, "CLCT"] = 0.0

    return X_train, X_test


@log_running_time
def export_data(folder: str, WF: str, X_train, X_test, y_train, sets) -> None:
    """Export data frames to csv.

    Args:
        folder: the folder where the csv files will be saved.
        WF: Wind Farm identification.
        X,y/train/test: data sets to be saved.

    Returns:
        None.
    """
    os.makedirs(folder + WF, exist_ok=True)
    y_test = sets.get("y_test")

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
    y_test.to_csv(folder + "{}/{}.csv".format(WF, "y_test"), index=False, header=False)
