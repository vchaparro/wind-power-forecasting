from kedro.pipeline import Pipeline, node
from typing import Dict
from .nodes import (
    get_data_by_wf,
    add_new_cols,
    input_missing_values,
    select_best_NWP_features,
    clean_outliers,
    fix_negative_clct,
    split_data_by_date,
    export_data,
)


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=get_data_by_wf,
                inputs=["X_train_raw", "y_train_raw", "params:wf"],
                outputs=["X_by_WF", "y_by_WF"],
                name="filter_data_by_WF",
            ),
            node(
                func=add_new_cols,
                inputs=["params:new_cols", "X_by_WF"],
                outputs=["X_by_WF_newcols", "cols"],
                name="add_new_columns",
            ),
            node(
                func=input_missing_values,
                inputs=["X_by_WF_newcols", "cols", "params:cols_to_interpol"],
                outputs="X_no_missing_vals",
                name="input_missing_values",
            ),
            node(
                func=select_best_NWP_features,
                inputs=["X_no_missing_vals"],
                outputs="X_with_best_NWP",
                name="select_best_NWP_features",
            ),
            node(
                func=clean_outliers,
                inputs=["X_with_best_NWP", "y_by_WF", "parameters"],
                outputs=["X_cleaned", "y_cleaned"],
                name="clean_outliers",
            ),
            node(
                func=fix_negative_clct,
                inputs=["X_cleaned"],
                outputs=None,
                name="fix_negative_CLCT",
            ),
            node(
                func=split_data_by_date,
                inputs=["params:split_date", "X_cleaned", "y_cleaned"],
                outputs="splitted_data_sets",
                name="split_data",
            ),
            node(
                func=export_data,
                inputs=["params:folder.pri", "params:wf", "splitted_data_sets"],
                outputs=None,
                name="export_data_to_csv",
            ),
        ],
        tags="data_preparation_for_ML",
    )
