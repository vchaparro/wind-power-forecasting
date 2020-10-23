"""Construction of the master pipeline.
"""

from typing import Dict

from kedro.pipeline import Pipeline

from .pipelines.CNR_challenge import data_preprocess as dcnr
from .pipelines.CNR_challenge import feature_engineering as fecnr
from .pipelines.CNR_challenge import make_models as pcnr
from .pipelines.data_engineering import pipeline as de
from .pipelines.data_engineering.nodes import log_running_time
from .pipelines.exploratory_data_analysis import pipeline as eda
from .pipelines.feature_engineering import pipeline as fe
from .pipelines.modeling import pipeline as mdl
from .pipelines.neural_network import pipeline as nn


def create_pipelines(**kwargs) -> Dict[str, Pipeline]:
    """Create the project's pipeline.

    Args:
        kwargs: Ignore any additional arguments added in the future.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.

    """
    data_engineering = de.create_pipeline().decorate(log_running_time)
    exploratory_data_analysis = eda.create_pipeline().decorate(log_running_time)
    feature_engineering = fe.create_pipeline().decorate(log_running_time)
    modeling = mdl.create_pipeline().decorate(log_running_time)
    neural_network = nn.create_pipeline().decorate(log_running_time)
    data_for_CNR = dcnr.create_pipeline().decorate(log_running_time)
    feat_engineering_CNR = fecnr.create_pipeline().decorate(log_running_time)
    predictions_CNR = pcnr.create_pipeline().decorate(log_running_time)

    return {
        "de": data_engineering,
        "eda": exploratory_data_analysis,
        "fe": feature_engineering,
        "mdl": modeling,
        "nn": neural_network,
        "dcnr": data_for_CNR,
        "fecnr": feat_engineering_CNR,
        "pcnr": predictions_CNR,
        "__default__": data_engineering
        + feature_engineering
        + exploratory_data_analysis
        + modeling
        + neural_network
        + data_for_CNR
        + feat_engineering_CNR
        + predictions_CNR,
    }
