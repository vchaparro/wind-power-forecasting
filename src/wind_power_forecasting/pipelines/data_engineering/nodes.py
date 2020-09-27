import pandas as pd
from functools import wraps
from typing import Callable, Dict, List
import time
import logging
import numpy as np
from pathlib import Path
import datetime as dt
import os
import re
from metpy import calc
from metpy.units import units
import matplotlib.pyplot as plt
from operational_analysis.toolkits import filters
from operational_analysis.toolkits import power_curve
from kedro.framework import context


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
    """ Function to get wind speed from wind velocity components U and V.
    
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
    fig_id: int,
    folder: str,
    WF: str,
    tight_layout=True,
    fig_extension="png",
    resolution=300,
):
    os.makedirs(folder + WF, exist_ok=True)
    path = os.path.join(folder + WF, fig_id + "." + fig_extension)

    if tight_layout:
        plt.tight_layout()

    plt.savefig(path, format=fig_extension, dpi=resolution)


def _plot_flagged_pc(ws, p, flag_bool, alpha):
    plt.scatter(ws, p, s=3, alpha=alpha)
    plt.scatter(ws[flag_bool], p[flag_bool], s=3, c="red")
    plt.xlabel("Wind speed (m/s)")
    plt.ylabel("Power (MWh)")


def get_data_by_wf(X: pd.DataFrame, y: pd.DataFrame, wf: str) -> pd.DataFrame:
    """ Get data filterd by Wind Farm (wf).

        Args:
            X: X_train_raw.
            y: y_train_raw. 
            wf: Wind Farm identification.
            
        Returns:
            X, y data frames filtered by Wind Farm.
    """

    # Row selection by WF
    X = X[X["WF"] == wf]
    X["Time"] = pd.to_datetime(X["Time"], format="%d/%m/%Y %H:%M")

    # Save observations identification
    ID_X = X["ID"]

    # selecting rows of y_train and y_test
    y = y["Production"]
    y = y.loc[ID_X.values - 1]

    return X, y


def add_new_cols(new_cols: list, X: pd.DataFrame) -> pd.DataFrame:
    """ Adds new columns to a given data frame.
    
        Args: 
            new_cols: List with the column names to be added.
            X: data frame that will be expanded with new_cols.
            
        Returns:
            X expanded with the new columns and the columns that 
            contains missing values.
            
    """
    cols = X.columns[3:]
    for col in new_cols:
        X[col] = np.nan

    return X, cols


def _interpolate_missing_values(X: pd.DataFrame, cols: List) -> pd.DataFrame:
    """ Imputes those missing values due to the NWP's frequency in data providing.

        Args:
            X: Data frame where missing values are being inputed. 
            cols: list of column names of the data frame that have missing values.
            
        Returns:
            None, missing values are inputed inplace.
             
    """
    X.index = X["Time"]
    del X["Time"]

    for var in cols:
        X[var].interpolate(method="time", inplace=True, limit=2, limit_direction="both")

    X.reset_index(inplace=True)


@log_running_time
def input_missing_values(
    X: pd.DataFrame, cols: List, cols_to_interpol: List
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

    # Interpolate missing values when required.
    _interpolate_missing_values(X, cols_to_interpol)

    return X


def select_best_NWP_features(X: pd.DataFrame) -> pd.DataFrame:
    """ Select the features of the best NWP.

        Args:
            X: features data frame.
            
        Returns:
            Data frame with the best NWP features.

    """
    # Select the best NWP predictions for weather predictors
    X["U"] = X.NWP1_U
    X["V"] = X.NWP1_V
    X["T"] = X.NWP3_T
    X["CLCT"] = X.NWP4_CLCT

    # Select final features
    X = X[["ID", "Time", "U", "V", "T", "CLCT"]]

    return X


def _find_outliers(X: pd.DataFrame, y: pd.Series, wf: str) -> Dict[str, list]:
    """ Finds outliers based on power curve, using a binning method.
    
        Args:
            X: Feature data frame.
            y: target.
            parameters: dictionary containing the configuration parameters.
            
        Returns:
            outliers: dictionary with the different type of outliers found.

    """
    # Loading context
    ctx = context.load_context("../wind-power-forecasting")

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
    max_bin = 0.97 * X_["Production"].max()
    sparse = filters.bin_filter(
        X_.Production,
        X_.wspeed,
        sparse_bin_width,
        frac_std * X_.Production.std(),
        "median",
        0.1,
        max_bin,
        threshold_type,
        "all",
    )

    # bottom-curve stacked outliers
    bottom = filters.window_range_flag(
        X_.wspeed, bottom_max, 40, X_.Production, 0.05, 2000.0
    )

    # Plot outliers
    _plot_flagged_pc(X_.wspeed, X_.Production, np.repeat("True", len(X_.wspeed)), 0.3)
    _plot_flagged_pc(
        X_.wspeed, X_.Production, (top) | (sparse) | (bottom), 0.3,
    )
    _save_fig(
        "outliers",
        ctx.params.get("folder").get("rep") + "figures/",
        ctx.params.get("wf"),
    )
    plt.close()

    # Populate the dictionary
    outliers["top"] = top
    outliers["sparse"] = sparse
    outliers["bottom"] = bottom

    del X["Production"], X["wspeed"], X_

    return outliers


@log_running_time
def clean_outliers(X: pd.DataFrame, y: pd.Series, wf: str) -> pd.DataFrame:
    """ It removes the outliers and returned cleaned X, y.
    
        Args:
            X: the feature data frame.
            y: the target.
            
        Returns:
            Cleaned X, y.
            
    """
    # Find outliers
    outliers = _find_outliers(X, y, wf)

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


def fix_negative_clct(X: pd.DataFrame) -> None:
    """ Replaces negative values of CLCT by 0.
    
        Args:
            df: the data frame containing CLCT column.
            
        Returns:
            None, it replaces the values inplace.
    
    """
    X.loc[X["CLCT"] < 0, "CLCT"] = 0.0


def split_data_by_date(date: str, X: pd.DataFrame, y: pd.Series) -> Dict:
    """ It splits X and y sets by a 'Time' value  into sets for training and testing. 

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

    return sets


@log_running_time
def export_data(folder: str, WF: str, df_dict: Dict,) -> None:
    """ Export data frames to csv.

        Args: 
            folder: the folder where the csv files will be saved.
            WF: Wind Farm identification.
            df_dict: a dictionary with key, value pairs df name, df values.
            
        Returns:
            None.
    """
    os.makedirs(folder + WF, exist_ok=True)
    for key, value in df_dict.items():
        value.to_csv(
            folder + "{}/{}.csv".format(WF, key),
            index=False,
            date_format="%d/%m/%Y %H:%M",
        )

