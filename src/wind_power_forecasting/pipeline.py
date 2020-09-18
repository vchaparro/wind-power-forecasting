# Copyright 2020 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#


"""Construction of the master pipeline.
"""

from typing import Dict
from .pipelines.data_engineering import pipeline as de
from .pipelines.data_engineering.nodes import log_running_time
from .pipelines.exploratory_data_analysis import pipeline as eda
from .pipelines.feature_engineering import pipeline as fe
from .pipelines.modeling import pipeline as mdl
from kedro.pipeline import Pipeline


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

    return {
        "de": data_engineering,
        "eda": exploratory_data_analysis,
        "fe": feature_engineering,
        "mdl": modeling,
        "__default__": exploratory_data_analysis
        + data_engineering
        + feature_engineering
        + modeling,
    }
