"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.1
"""
from typing import Tuple

from keras.datasets import mnist
import numpy as np


def load_mnist() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loading mnist dataset.

    Args:
        None.
    Returns:
        Raw dataset.
        Each data is returned as numpy.ndarray type.
    """
    (train_inputs, train_labels), (test_inputs, test_labels) = mnist.load_data()

    return train_inputs, train_labels, test_inputs, test_labels


def preprocess_mnist(images: np.ndarray) -> np.ndarray:
    """Preprocesses the image data for inputs.

    Args:
        images: Raw mnist image data.
    Returns:
        Preprocessed data, with reshaping as a 2d array and normalized each value in the range of [0,1].
    """
    data_shape = images.shape

    preprocessed_inputs = images.reshape(
        (data_shape[0], data_shape[1]*data_shape[2])
    ).astype('float32')/255

    return preprocessed_inputs
