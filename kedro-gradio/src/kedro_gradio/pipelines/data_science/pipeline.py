"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.1
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import train_model, evaluate_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=train_model,
            inputs=["train_inputs", "train_labels", "params:model_options"],
            outputs="trained_model",
            name="train_model_node",
        ),
        node(
            func=evaluate_model,
            inputs=["trained_model", "test_inputs", "test_labels"],
            outputs=None,
            name="evaluate_model_node",
        ),
    ])
