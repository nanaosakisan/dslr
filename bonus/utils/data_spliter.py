import numpy as np
import random

from typing import Tuple


def data_spliter(
    x: np.ndarray, y: np.ndarray, proportion: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shuffles and splits the dataset (given by x and y) into a training and a test set,
    while respecting the given proportion of examples to be kept in the training set.
    Args:
        x: has to be an numpy.array, a matrix of shape m * n.
        y: has to be an numpy.array, a vector of shape m * 1.
        proportion: has to be a float, the proportion of the dataset that will be assigned to the
        training set.
    Return:
        (x_train, x_test, y_train, y_test) as a tuple of numpy.array
        None if x or y is an empty numpy.array.
        None if x and y do not share compatible shapes.
        None if x, y or proportion is not of expected type.
    Raises:
        This function should not raise any Exception.
    """
    if (
        not isinstance(x, np.ndarray)
        or not isinstance(y, np.ndarray)
        or not isinstance(proportion, float)
        or len(x) == 0
        or len(y) == 0
        or proportion < 0
        or proportion > 1
        or x.shape[0] != y.shape[0]
        or y.shape[1] != 1
    ):
        return None
    nb_elem_train = int(x.shape[0] * proportion)
    id_elem_train = random.sample(range(x.shape[0]), nb_elem_train)
    return (
        x[id_elem_train],
        np.delete(x, id_elem_train, 0),
        y[id_elem_train],
        np.delete(y, id_elem_train, 0),
    )
