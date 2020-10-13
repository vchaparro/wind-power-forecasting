from kedro.pipeline import Pipeline, node
from typing import Dict
from .nodes import (
    get_data_by_wf,
    add_new_cols,
    input_missing_values,
    select_best_NWP_features,
    clean_outliers,
    fix_negative_values,
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
                func=split_data_by_date,
                inputs=["params:split_date", "X_by_WF", "y_by_WF"],
                outputs="splitted_data_sets",
                name="split_data",
            ),
            node(
                func=add_new_cols,
                inputs=["params:new_cols", "splitted_data_sets"],
                outputs=["X_train_ncols", "X_test_ncols"],
                name="add_new_columns",
            ),
            node(
                func=input_missing_values,
                inputs=["X_train_ncols", "X_test_ncols"],
                outputs=["X_train_no_nans", "X_test_no_nans"],
                name="input_missing_values",
            ),
            node(
                func=select_best_NWP_features,
                inputs=["X_train_no_nans", "X_test_no_nans"],
                outputs=["X_train_bst_NWP", "X_test_bst_NWP"],
                name="select_best_NWP_features",
            ),
            node(
                func=clean_outliers,
                inputs=["X_train_bst_NWP", "splitted_data_sets", "params:wf"],
                outputs=["X_train_clnd", "y_train_clnd"],
                name="clean_outliers",
            ),
            node(
                func=fix_negative_values,
                inputs=["X_train_clnd", "X_test_bst_NWP"],
                outputs=["X_train_fixed", "X_test_fixed"],
                name="fix_negative_values",
            ),
            node(
                func=export_data,
                inputs=[
                    "params:folder.pri",
                    "params:wf",
                    "X_train_fixed",
                    "X_test_fixed",
                    "y_train_clnd",
                    "splitted_data_sets",
                ],
                outputs=None,
                name="export_data_to_csv",
            ),
        ],
        tags="data_preparation_for_ML",
    )
