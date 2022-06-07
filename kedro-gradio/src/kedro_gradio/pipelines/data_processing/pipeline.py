"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.1
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import load_mnist, preprocess_mnist


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=load_mnist,
            inputs=None,
            outputs=["raw_train_inputs", "train_labels", "raw_test_inputs", "test_labels"],
            name="load_mnist_data_node",
        ),
        node(
            func=preprocess_mnist,
            inputs="raw_train_inputs",
            outputs="train_inputs",
            name="preprocess_training_data_node",
        ),
        node(
            func=preprocess_mnist,
            inputs="raw_test_inputs",
            outputs="test_inputs",
            name="preprocess_test_data_node",
        ),
    ])
