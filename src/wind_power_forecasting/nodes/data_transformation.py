import pandas as pd
import numpy as np
import datetime as dt
import re
from collections import OrderedDict
from sklearn.base import BaseEstimator, TransformerMixin
from metpy import calc
from metpy.units import units


def get_wind_speed(x: pd.DataFrame) -> float:
    """ Function to get wind speed from wind velocity components U and V.
    
        Args:
            x: Features data frame containing components U and V as columns.
            
        Returns: 
            Wind speed for each pair <U,V>.

    """
    return float(
        calc.wind_speed(
            x.U * units.meter / units.second, x.V * units.meter / units.second
        ).magnitude
    )


def get_wind_dir(x: pd.DataFrame) -> float:
    """ Function to get wind direction from wind velocity components U and V.
    
        Args:
            x: Features data frame containing components U and V as columns.
            
        Returns: 
            Wind direction for each pair <U,V>.

    """

    return float(
        calc.wind_direction(
            x.U * units.meter / units.second,
            x.V * units.meter / units.second,
            convention="from",
        ).magnitude
    )


# Enconding cyclic variables
def encode_cyclic(data, col, max_val):
    data[col + "_sin"] = np.sin(2 * np.pi * data[col] / max_val)
    data[col + "_cos"] = np.cos(2 * np.pi * data[col] / max_val)


# Enconding cyclic variables
def encode_cyclic(data, col, max_val):
    data[col + "_sin"] = np.sin(2 * np.pi * data[col] / max_val)
    data[col + "_cos"] = np.cos(2 * np.pi * data[col] / max_val)


# Class to add the new features
class NewFeaturesAdder(BaseEstimator, TransformerMixin):
    """ Scikit-learn custom transformer that allows to add new features 
        derived from the original ones.
    """

    def __init__(
        self,
        add_time_feat=False,
        add_cycl_feat=False,
        add_inv_T=False,
        add_interactions=False,
    ):
        self.add_time_feat = add_time_feat
        self.add_cycl_feat = add_cycl_feat
        self.add_inv_T = add_inv_T
        self.add_interactions

    def fit(self, documents, y=None):
        return self

    def transform(self, x_dataset):

        # Velocity derived features
        x_dataset["wspeed"] = x_dataset.apply(_get_wind_speed, axis=1)
        x_dataset["wdir"] = x_dataset.apply(_get_wind_dir, axis=1)

        if self.add_interactions:
            x_dataset["wspeed_wdir"] = x_dataset["wspeed"] * x_dataset["wdir"]

            if self.add_inv_T:
                x_dataset["wspeed_invT"] = x_dataset["wspeed"] * x_dataset["inv_T"]
                x_dataset["wspeed_wdir_invT"] = (
                    x_dataset["wspeed"] * x_dataset["wdir"] * x_dataset["inv_T"]
                )

        if self.add_time_feat:
            # Time derived features
            x_dataset["hour"] = x_dataset["Time"].dt.hour
            x_dataset["month"] = x_dataset["Time"].dt.month

        if self.add_cycl_feat:
            if self.add_time_feat == False:
                _encode_cyclic(x_dataset, "wdir", 360)
            else:
                # Hour
                _encode_cyclic(x_dataset, "hour", 24)

                # Month
                _encode_cyclic(x_dataset, "month", 12)

                # Wind direction
                _encode_cyclic(x_dataset, "wdir", 360)

        if self.add_inv_T:
            x_dataset["inv_T"] = 1 / x_dataset["T"]

        return x_dataset


def get_col_prefixes(cols, regex):

    prefix_lst = []
    p = re.compile(regex)

    for col in cols:
        m = p.match(col)

        if m is not None:
            col_prefix = "NWP" + m.group("NWP") + "_" + m.group("met_var")
            prefix_lst.append(col_prefix)

    prefix_lst = list(OrderedDict.fromkeys(prefix_lst))

    return prefix_lst


def get_df_for_eda(df, regex=r"NWP(?P<NWP>\d{1})_(?P<run>\d{2}h)_(?P<fc_day>D\W?\d?)_"):
    """
        Convert the dataframe (test/train) to an easily manipulate format,
        without changing the data itself.      
    """
    # Create a temporal dataframe
    df_tmp = pd.DataFrame([])

    # Regular expresion to capture the values from the column names
    p = re.compile(regex)

    # Get prefix list
    cols = df.columns[3:-1]
    prefix_lst = get_col_prefixes(cols, regex)

    for prefix in prefix_lst:

        # Create a second temporal dataframe
        df_tmp2 = pd.DataFrame(
            np.nan,
            index=df.index,
            columns=[
                "WF",
                "NWP",
                "fc_day",
                "run",
                "id_target",
                "time",
                "U",
                "V",
                "T",
                "CLCT",
                "production",
            ],
        )

        # Get values using the regex
        m = p.match(prefix)

        # Select df columns that start with col_prefix
        sub_df = df.filter(regex="^" + prefix, axis=1)

        # Populate
        df_tmp2["WF"] = df["WF"]
        df_tmp2["NWP"] = m.group("NWP")
        df_tmp2["FC_Day"] = m.group("fc_day")
        df_tmp2["Run"] = m.group("run")
        df_tmp2["ID_target"] = df["ID"]
        df_tmp2["Time"] = df["Time"]

        # Some of these weather parameters may no exist for every column
        try:
            df_tmp2["U"] = sub_df[prefix + "U"]
        except KeyError:
            pass

        try:
            df_tmp2["V"] = sub_df[prefix + "V"]
        except KeyError:
            pass

        try:
            df_tmp2["T"] = sub_df[prefix + "T"]
        except KeyError:
            pass

        try:
            df_tmp2["CLCT"] = sub_df[prefix + "CLCT"]
        except KeyError:
            pass

        # Just in case there's not 'Production' column (f.i., X_test)
        if "Production" in df.columns:
            df_tmp2["Production"] = df["Production"]
        else:
            df_tmp2["Production"] = np.nan

        df_tmp = df_tmp.append(df_tmp2, ignore_index=True)
        del df_tmp2

    return df_tmp


def split_data_by_date(date, X, y):
    """
    It splits X and y sets by a 'Time' value 
    into sets for training and testing. 
        - Return: a dictionary with the four sets
                  (X_train, y_train, X_test, y_test)
    """
    sets = {}
    date_cut = dt.datetime.strptime(date, "%Y-%m-%d %H:%M:%S")

    X_test = X[X["Time"] > date_cut]
    X_train = X[X["Time"] <= date_cut]
    y_train = y[y.ID.isin(X_train.ID)]
    y_test = y[y.ID.isin(X_test.ID)]

    sets["X_train"] = X_train
    sets["X_test"] = X_test
    sets["y_train"] = y_train
    sets["y_test"] = y_test

    return sets


def add_new_cols(new_cols, df):

    for col in new_cols:
        df[col] = np.nan


def input_missing_values(df, cols):

    regex = (
        "NWP(?P<NWP>\d{1})_(?P<run>\d{2}h)_(?P<fc_day>D\W?\d?)_(?P<weather_var>\w{1,4})"
    )
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
                    df["NWP" + key + "_" + i] = df["NWP" + key + "_" + i].fillna(
                        df[col_name]
                    )

    return df


def interpolate_missing_values(df, cols, index):

    df.index = df[index]
    del df[index]

    for var in cols:
        df[var].interpolate(
            method="time", inplace=True, limit=2, limit_direction="both"
        )

    df.reset_index(inplace=True)

    return df


def add_wind_vars(df, regex):

    p = re.compile(regex)
    cols = get_col_prefixes(list(df.columns), regex)

    for col in cols:
        m = p.match(col)

        nwp = m.group("NWP")
        sub_df = df[["NWP" + nwp + "_" + "U", "NWP" + nwp + "_" + "V"]]
        sub_df.rename(
            columns={"NWP" + nwp + "_" + "U": "U", "NWP" + nwp + "_" + "V": "V"},
            inplace=True,
        )

        df["NWP" + nwp + "_wvel"] = sub_df.apply(get_wind_velmod, axis=1)
        df["NWP" + nwp + "_wdir"] = sub_df.apply(get_wind_dir, axis=1)
        df["NWP" + nwp + "_wdir_sin"] = np.sin(
            2 * np.pi * df["NWP" + nwp + "_wdir"] / 360
        )
        df["NWP" + nwp + "_wdir_cos"] = np.cos(
            2 * np.pi * df["NWP" + nwp + "_wdir"] / 360
        )

        if nwp == "4":
            height = 10
            df["NWP" + nwp + "_wshear"] = df["NWP" + nwp + "_wvel"] * (
                (50 / height) ** 0.14
            )
        else:
            height = 100
            df["NWP" + nwp + "_wshear"] = df["NWP" + nwp + "_wvel"] * (
                (50 / height) ** 0.14
            )

        del sub_df

    return df


def add_time_vars(df, time_col):

    df["hour"] = df[time_col].dt.hour
    df["month"] = df[time_col].dt.month

    encode_cyclic(df, "hour", 24)
    encode_cyclic(df, "month", 12)

    return df


def convert_to_celsius(df, col_list):

    for col in col_list:
        df[col] = df[col] - 273.15

    return df
