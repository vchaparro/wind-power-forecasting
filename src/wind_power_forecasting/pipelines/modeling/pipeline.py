from kedro.pipeline import Pipeline, node

from .nodes import create_model, get_model_plots, test_model


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=create_model,
                inputs=[
                    "params:wf",
                    "params:alg",
                    "params:find_max_kbests",
                    "params:max_k_bests",
                    "params:n_splits",
                    "params:transform_target",
                    "params:scoring",
                    "params:refit",
                ],
                outputs=[
                    "wf",
                    "alg",
                    "best_model",
                    "X_train",
                    "y_train",
                ],
                name="create_model",
            ),
            node(
                func=test_model,
                inputs=[
                    "alg",
                    "wf",
                    "best_model",
                    "X_train",
                    "y_train",
                    "params:folder.mdl",
                    "params:transform_target",
                ],
                outputs=[
                    "model",
                    "X_train_2",
                    "X_test_2",
                    "predictions",
                ],
                name="test_model",
            ),
            node(
                func=get_model_plots,
                inputs=[
                    "wf",
                    "alg",
                    "model",
                    "predictions",
                    "X_train_2",
                    "X_test_2",
                    "params:n_splits",
                    "params:val_score_plot",
                    "params:folder.rep",
                ],
                outputs=None,
                name="get_model_plots",
            ),
        ],
        tags="modeling",
    )
