from kedro.pipeline import Pipeline, node
from .nodes import train_test_ann
from .nodes import get_nn_plots


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=train_test_ann,
                inputs=[
                    "params:wf",
                    "params:n_splits",
                    "params:find_max_kbests",
                    "params:max_k_bests",
                    "params:transform_target",
                    "params:scoring",
                    "params:refit",
                ],
                outputs=["X_train_2_nn", "X_test_2_nn", "predictions_nn"],
                name="train_test_ann",
            ),
            node(
                func=get_nn_plots,
                inputs=[
                    "params:wf",
                    "params:alg",
                    "predictions_nn",
                    "X_train_2_nn",
                    "X_test_2_nn",
                    "params:n_splits",
                    "params:val_score_plot",
                    "params:folder.rep",
                ],
                outputs=None,
                name="get_nn_plots",
            ),
        ],
        tags="neural_network",
    )

