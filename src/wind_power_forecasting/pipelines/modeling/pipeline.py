from kedro.pipeline import Pipeline, node
from .nodes import create_model, train_test_model, get_model_plots


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
                    "params:mars_hypms",
                ],
                outputs=[
                    "wf",
                    "algorithm",
                    "gcv",
                    "X_train",
                    "y_train",
                    "X_test",
                    "y_test",
                    "cape_scorer",
                ],
                name="create_model",
            ),
            node(
                func=train_test_model,
                inputs=[
                    "algorithm",
                    "wf",
                    "gcv",
                    "X_train",
                    "y_train",
                    "params:folder.mdl",
                ],
                outputs=["model", "X_train_2", "X_test_2"],
                name="train_test_model",
            ),
            node(
                func=get_model_plots,
                inputs=[
                    "wf",
                    "algorithm",
                    "model",
                    "X_train",
                    "y_train",
                    "X_test",
                    "y_test",
                    "X_train_2",
                    "X_test_2",
                    "params:n_splits",
                    "cape_scorer",
                    "params:folder.rep",
                ],
                outputs=None,
                name="get_model_plots",
            ),
        ],
        tags="modeling",
    )

