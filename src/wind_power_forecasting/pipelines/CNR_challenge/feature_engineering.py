from kedro.pipeline import Pipeline, node
from ..feature_engineering.nodes import (
    feature_engineering,
    save_prepared_data,
)


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=feature_engineering,
                inputs=[
                    "params:wf",
                    "params:folder.cnr.primary",
                    "params:add_time_feat",
                    "params:add_cycl_feat",
                    "params:add_inv_T",
                    "params:add_interactions",
                ],
                outputs=["Xtrain_pped", "Xtest_pped", "feat_names",],
                name="feat_engineering",
            ),
            node(
                func=save_prepared_data,
                inputs=[
                    "params:folder.cnr.prepared",
                    "Xtrain_pped",
                    "Xtest_pped",
                    "feat_names",
                    "params:wf",
                ],
                outputs=None,
                name="save_pped_data",
            ),
        ],
        tags="feature_engineering_CNR",
    )
