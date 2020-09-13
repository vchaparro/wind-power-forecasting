from kedro.pipeline import Pipeline, node
from .nodes import create_model


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=create_model,
                inputs=[
                    "params:wf",
                    "params:algorithm",
                    "params:k_bests",
                    "params:n_splits",
                    "X_train_pped_WF1",
                    "y_train_WF1",
                    "feature_names_WF1",
                    "params:mars",
                ],
                outputs="gcv",
                name="create_model",
            ),
        ],
        tags="modeling",
    )

