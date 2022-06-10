"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.1
"""
import logging
from typing import Dict

import tensorflow as tf


def build_model(input_shape: int, class_size: int) -> tf.keras.Model:
    """
    A function to build a simple neural network model.
    Args:
        input_shape: Number of input dimension.
        class_size: Number of output dimension.
    Returns:
        Tensorflow model.
    """
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(input_shape,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(class_size, activation='softmax')
    ])
    return model

def train_model(train_inputs, train_labels, parameters: Dict) -> tf.keras.Model:
    """
    A function to prepair the model and train with the training dataset.
    After training process, model will be saved as pickle format.
    Params: Parameters for training variables like batch_size, epochs, val_size, shuffle.
    Return: Trained model
    """

    logger = logging.getLogger(__name__)
    logger.info(f"Dataset shape: {train_inputs.shape}.")

    model = build_model(train_inputs.shape[1], len(set(train_labels)))
    model.compile(
        optimizer='adam', 
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        x=train_inputs, 
        y=train_labels, 
        batch_size=parameters["batch_size"],
        epochs=parameters["epochs"],
        validation_split=parameters["val_size"],
        shuffle=parameters["shuffle"],
    )

    return model


def evaluate_model(model, test_inputs, test_labels) -> None:
    """
    A function to evaluate the trained model with the test dataset.
    Args:
        model: Trained tensorflow model.
        test_inputs: Test input data.
        test_labels: Test output data.
    Returns:
        None
    """
    test_loss, test_acc = model.evaluate(
        test_inputs,  
        test_labels, 
        verbose=2
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Accuracy: {test_acc}, Loss: {test_loss} on the test data.")
