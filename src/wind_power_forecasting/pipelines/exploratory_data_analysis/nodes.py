import datetime as dt
import logging
import os
import re
import time
from functools import wraps
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import pandas as pd


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


@log_running_time
def build_df_for_eda(X: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    """ Converts raw data frames to a convenient format for EDA, without changing the data itself.

        Args:
            X: X_train_raw.
            y: y_train_raw.
            
        Returns:
            Data frame ready for EDA.

    """
    X["Time"] = pd.to_datetime(X["Time"], format="%d/%m/%Y %H:%M")
    X["Production"] = y["Production"]

    # regex that fits the column names of X_train_raw.
    regex = r"NWP(?P<NWP>\d{1})_(?P<run>\d{2}h)_(?P<fc_day>D\W?\d?)_"

    # Create a temporal dataframe
    df_tmp = pd.DataFrame([])

    # Regular expresion to capture the values from the column names
    p = re.compile(regex)

    # Get prefix list
    cols = X.columns[3:-1]

    for col in cols:

        # Create a second temporal dataframe
        df_tmp2 = pd.DataFrame(
            np.nan,
            index=X.index,
            columns=[
                "WF",
                "NWP",
                "fc_day",
                "run",
                "id",
                "time",
                "U",
                "V",
                "T",
                "CLCT",
                "production",
            ],
        )

        # Get values using the regex
        m = p.match(col)
        prefix = (
            "NWP"
            + m.group("NWP")
            + "_"
            + m.group("run")
            + "_"
            + m.group("fc_day")
            + "_"
        )

        # Populate
        df_tmp2["WF"] = X["WF"]
        df_tmp2["NWP"] = m.group("NWP")
        df_tmp2["fc_day"] = m.group("fc_day")
        df_tmp2["run"] = m.group("run")
        df_tmp2["id"] = X["ID"]
        df_tmp2["time"] = X["Time"]

        # Some of these weather parameters may not exist for every column
        try:
            df_tmp2["U"] = X[prefix + "U"]
        except KeyError:
            pass

        try:
            df_tmp2["V"] = X[prefix + "V"]
        except KeyError:
            pass

        try:
            df_tmp2["T"] = X[prefix + "T"]
        except KeyError:
            pass

        try:
            df_tmp2["CLCT"] = X[prefix + "CLCT"]
        except KeyError:
            pass

        # Just in case there's not 'Production' column (f.i., X_test)
        if "Production" in X.columns:
            df_tmp2["production"] = X["Production"]
        else:
            df_tmp2["production"] = np.nan

        df_tmp = df_tmp.append(df_tmp2, ignore_index=True)
        del df_tmp2

    return df_tmp


@log_running_time
def get_data_by_wf(df: pd.DataFrame, folder: str) -> List:
    """ Filters the data by WF and exports to csv files.

        Args:
            df: A data frame in the proper format for EDA.
            
        Returns:
            A list of data frames, one for each Wind Farm.

    """

    df_lst = []
    wf_lst = df["WF"].unique()
    os.makedirs(folder + "/for_EDA_by_WF", exist_ok=True)

    for wf in wf_lst:
        df_lst.append(df[df["WF"] == wf])

    for i in range(len(df_lst)):
        del df_lst[i]["WF"]
        df_lst[i].to_csv(
            folder + "for_EDA_by_WF/df_WF{}.csv".format(i + 1),
            index=False,
            date_format="%d/%m/%Y %H:%M",
        )

    return df_lst
