from kedro.pipeline import Pipeline, node
from .nodes import (
    get_data_by_wf,
    add_new_cols,
    input_missing_values,
    fix_negative_values,
    export_data,
    select_best_NWP_features,
    clean_outliers,
)


def create_pipeline(**kargs):
    return Pipeline(
        [
            node(
                func=get_data_by_wf,
                inputs=["params:wf", "X_train_raw", "X_test_raw", "y_train_raw"],
                outputs=["X_train_by_WF", "X_test_by_WF", "y_train_by_WF"],
                name="select_data_by_WF",
            ),
            node(
                func=add_new_cols,
                inputs=["params:new_cols", "X_train_by_WF", "X_test_by_WF"],
                outputs=[
                    "X_train_newcols",
                    "X_test_newcols",
                    "cols_train",
                    "cols_test",
                ],
                name="new_columns",
            ),
            node(
                func=input_missing_values,
                inputs=[
                    "X_train_newcols",
                    "X_test_newcols",
                    "cols_train",
                    "cols_test",
                    "params:cols_to_interpol",
                ],
                outputs=["X_train_no_missings", "X_test_no_missings"],
                name="missing_values_input",
            ),
            node(
                func=select_best_NWP_features,
                inputs=["X_train_no_missings", "X_test_no_missings"],
                outputs=["X_train_best_NWP", "X_test_best_NWP"],
                name="best_NWP_vars",
            ),
            node(
                func=clean_outliers,
                inputs=[
                    "X_train_best_NWP",
                    "y_train_by_WF",
                    "params:wf",
                    "params:outliers_plot_id",
                ],
                outputs=["X_train_cleaned", "y_train_cleaned"],
                name="remove_outliers",
            ),
            node(
                func=fix_negative_values,
                inputs=["X_train_cleaned", "X_test_best_NWP", "y_train_cleaned"],
                outputs="processed_data",
                name="replace_negative_values",
            ),
            node(
                func=export_data,
                inputs=["params:folder.cnr.primary", "params:wf", "processed_data"],
                outputs=None,
                name="save_to_csv",
            ),
        ],
        tags="prepare_data_CNR",
    )

