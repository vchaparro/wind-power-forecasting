from kedro.pipeline import Pipeline, node
from typing import Dict
from .nodes import build_df_for_eda, get_data_by_wf


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=build_df_for_eda,
                inputs=["X_train_raw", "y_train_raw"],
                outputs="df_for_eda",
                name="build_df_for_eda",
            ),
            node(
                func=get_data_by_wf,
                inputs=["df_for_eda", "params:folder.pri"],
                outputs="df_list_for_eda_by_WF",
                name="get_data_by_WF",
            ),
        ],
        tags="data_preparation_for_EDA",
    )

