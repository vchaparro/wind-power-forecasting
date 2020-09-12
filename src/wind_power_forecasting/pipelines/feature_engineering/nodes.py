import pandas as pd
from typing import List, Dict
import logging
import numpy as np
import datetime as dt
import os
import re
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.constants import convert_temperature
from metpy import calc
from metpy.units import units
import pickle
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.pipeline import Pipeline
from collections import OrderedDict
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer


def _get_wind_speed(x: pd.DataFrame) -> float:
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


def _get_wind_dir(x: pd.DataFrame) -> float:
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


def _encode_cyclic(data: pd.DataFrame, col: str, max_val: int) -> None:
    """ Encondes cyclic features using sinusoidal functions.

        Args:
            data: the data frame containing the cyclic features.
            col: the feature name to be enconded.
            max_val: the maximum value in each case (for instance: 24 for hour, 12 for month.)

        Returns:
            None, it performs the enconding inplace.

    """
    data[col + "_sin"] = np.sin(2 * np.pi * data[col] / max_val)
    data[col + "_cos"] = np.cos(2 * np.pi * data[col] / max_val)


def _temperature_in_celsius(df, temp_col):
    """ Converts temperature units from Kelvin to Celsius.

        Args:
            df: the data frame containing a temperature feature.
            temp_col: the temperature feature name.

        Returns:
            The initial dataframe with the temperature feature in Celsius.
            
    """
    df[col] = pd.Series(
        convert_temperature(np.array(df[temp_col]), "Kelvin", "Celsius")
    )
    return df


def _add_interaction_terms(df, var_list):
    """ Adds the possible interaction terms from the list elements.

        Args:
            var_list: list of features.

        Returns:
            The data frame with the interaction terms as new columns.
            
    """
    print("Function to be implemented!")


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
        self.add_interactions = add_interactions

    def fit(self, documents, y=None):
        return self

    def transform(self, x_dataset):

        # Velocity derived features
        x_dataset["wspeed"] = x_dataset.apply(_get_wind_speed, axis=1)
        x_dataset["wdir"] = x_dataset.apply(_get_wind_dir, axis=1)

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

        if self.add_interactions:
            x_dataset["wspeed_wdir"] = x_dataset["wspeed"] * x_dataset["wdir"]

            if self.add_inv_T:
                x_dataset["wspeed_invT"] = x_dataset["wspeed"] * x_dataset["inv_T"]
                x_dataset["wspeed_wdir_invT"] = (
                    x_dataset["wspeed"] * x_dataset["wdir"] * x_dataset["inv_T"]
                )

        return x_dataset


def _walklevel(some_dir, level):
    """
       It works just like os.walk, but you can pass it a level parameter
       that indicates how deep the recursion will go.
       If depth is -1 (or less than 0), the full depth is walked.
    """
    some_dir = some_dir.rstrip(os.path.sep)
    assert os.path.isdir(some_dir)
    num_sep = some_dir.count(os.path.sep)

    for root, dirs, files in os.walk(some_dir):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)

        if num_sep + level <= num_sep_this:
            del dirs[:]


def _load_csv_files(loc: str, level=0, header=0) -> Dict:
    """ It loads all csv files in a location and returns a list of data frames 
        created from those files. 

        Args:
            loc: directory where the csv files are located.
            level: how deep the recursion will go in the directory. By default is 0.
    """
    df_dict = {}
    for dirname, _, filenames in _walklevel(loc, level):
        for filename in filenames:
            df_dict[filename] = pd.read_csv(dirname + filename, sep=",", header=header)

    return df_dict


def feature_engineering(
    wf: str,
    data_source: str,
    add_time_feat: bool,
    add_cycl_feat: bool,
    add_inv_T: bool,
    add_interactions: bool,
) -> (np.ndarray, np.ndarray, List):
    """ It performs feature engineering pipeline with these steps:
            - Add new features derived from original ones.
            - Delete innecesary features.
            - Apply a power transform + standarization.

        Args: 
            data_source: folder where the primary data is located.
            add_time_feat: (True/False) to add or not features derived from Time.
            add_cycle_feat: (True/False) whether to encode cyclic features or not.
            add_inv_T: (True/False) to add or not 1/T derived feature from T.
            add_interactions: (True/False) to add or not interaction terms.
    

        Returns:
            X_train and X_test pickle objects prepared for modeling.
            A list with the dropped unnecessary columns.

    """
    # Import data sets depending on provided Wind Farm (wf)
    folder = data_source + wf + "/"
    X_train = _load_csv_files(folder).get("X_train.csv")
    X_test = _load_csv_files(folder).get("X_test.csv")
    X_train["Time"] = pd.to_datetime(X_train["Time"], format="%d/%m/%Y %H:%M")
    X_test["Time"] = pd.to_datetime(X_test["Time"], format="%d/%m/%Y %H:%M")

    feat_adder = NewFeaturesAdder(
        add_time_feat=add_time_feat,
        add_cycl_feat=add_cycl_feat,
        add_inv_T=add_inv_T,
        add_interactions=add_interactions,
    )

    drop_lst = []
    if feat_adder.get_params().get("add_cycl_feat"):
        if feat_adder.get_params().get("add_inv_T"):
            drop_lst = ["ID", "Time", "U", "V", "wdir", "hour", "month", "T"]
        else:
            drop_lst = ["ID", "Time", "U", "V", "wdir", "hour", "month"]
    else:
        if feat_adder.get_params().get("add_inv_T"):
            drop_lst = ["ID", "Time", "U", "V", "T"]
        else:
            drop_lst = ["ID", "Time", "U", "V"]

    pre_process = ColumnTransformer(
        remainder="passthrough", transformers=[("drop_columns", "drop", drop_lst)]
    )

    feat_eng_pipeline = Pipeline(
        steps=[
            ("feature_adder", feat_adder),
            ("pre_processing", pre_process),
            (
                "powertransformer",
                PowerTransformer(method="yeo-johnson", standardize=True),
            ),
        ]
    )

    X_train_pped = feat_eng_pipeline.fit_transform(X_train)
    X_test_pped = feat_eng_pipeline.transform(X_test)

    # Get the names of the features
    feature_names = X_train.drop(drop_lst, axis=1).columns

    return X_train_pped, X_test_pped, feature_names


def save_prepared_data(
    folder: str, X_train, X_test, feature_names: List, WF: str
) -> None:
    """ Saves the prepared data after feature engineering.

        Args: 
            folder: the folder where the pickle objects will be saved.
            WF: Wind Farm identification.
            
        Returns:
            None.
    """
    os.makedirs(folder + WF, exist_ok=True)

    with open(folder + "{}/X_train_pped.pickle".format(WF), "wb") as handle:
        pickle.dump(X_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(folder + "{}/X_test_pped.pickle".format(WF), "wb") as handle:
        pickle.dump(X_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(folder + "{}/feature_names.pkl".format(WF), "wb") as handle:
        pickle.dump(feature_names, handle, protocol=pickle.HIGHEST_PROTOCOL)


def show_feature_importance(
    wf: str, data_src: str, X: np.ndarray, feature_names: List, k_best: str
):
    """ It finds the k most influecers features on target by performing  
        mutual information regression. 

        Args:
            wf: Wind Farm identification.
            data_sr: location of data files.
            X: features data set.
            feature_names: the name of the features in X.
            k: the number of features to return. 
            
        Returns:
            A descending ordered list of the k best features. 
            If k='all' the list can be considered as a feature importance list.
            
    """
    # Load target data (y_train)
    y = _load_csv_files(data_src + wf + "/", header=None).get("y_train.csv")
    y = y.to_numpy().reshape(-1)

    if k_best == "all":
        k = X.shape[1]
    else:
        k = int(k_best)

    # Perform MI regression to get the k-best features.
    selec_k_best = SelectKBest(mutual_info_regression, k=k)
    selec_k_best.fit(X, y)

    # Get the names of the k-best features
    fnames = feature_names
    mask = selec_k_best.get_support()
    scores = selec_k_best.scores_
    selected_feat = {}

    for bool, feature, score in zip(mask, fnames, scores):
        if bool:
            selected_feat[feature] = score

    sorted_sel_feat = {
        k: v
        for k, v in reversed(sorted(selected_feat.items(), key=lambda item: item[1]))
    }

    i = 1
    print("Feature importance in descending order: ")
    for k, v in sorted_sel_feat.items():
        print("{0}. {1}: {2:.2f}".format(i, k, v))
        i += 1

    logger = logging.getLogger(__name__)
    logger.info("Feature importance in descending order: {}".format(sorted_sel_feat))
