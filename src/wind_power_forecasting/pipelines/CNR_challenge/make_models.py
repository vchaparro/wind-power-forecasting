from kedro.pipeline import Pipeline, node

from .nodes import train_model, predict


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=train_model,
                inputs=["params:alg", "params:wf",],
                outputs="model_CNR",
                name="train_model_CNR",
            ),
            node(
                func=predict,
                inputs=[
                    "params:wf",
                    "model_CNR",
                    "params:folder.cnr.predictions",
                    "params:alg",
                ],
                outputs="predictions_CNR",
                name="make_predicitons",
            ),
        ],
        tags="predictions_for_CNR",
    )
