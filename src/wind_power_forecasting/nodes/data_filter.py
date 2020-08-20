import os
import pandas as pd
import numpy as np
import datetime as dt
import gc
import re

def get_data_by_wf(df: pd.DataFrame, wf: str) -> pd.DataFrame:
    """ Get data filterd by Wind Farm (wf).

        Args:
            df: data frame containing data for all or several Wind Farms.
        Returns:
            A data frame with the data only for the indicated Wind Farm.
    """

    return df[df["WF"] == wf]
