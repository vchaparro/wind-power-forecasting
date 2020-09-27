from kedro.pipeline import Pipeline, node
from .nodes import train_test_ann
from wind_power_forecasting.pipelines.modeling.nodes import get_model_plots


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
                outputs=[
                    "predictions_",
                    "X_train_2_",
                    "X_test_2_",
                    "cape_scorer_",
                ],
                name="train_test_ann",
            ),
        ],
        tags="neural_network",
    )

