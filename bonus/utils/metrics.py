import numpy as np

from typing import Tuple, Union, Optional


def count_tp_tn_fp_fn(
    y: np.ndarray, y_hat: np.ndarray, pos_label: Union[int, str] = 1
) -> Tuple[int, int, int, int]:
    tp, tn, fp, fn = 0, 0, 0, 0
    for yi, y_hati in zip(y, y_hat):
        if yi == y_hati == pos_label:
            tp += 1
        elif yi == y_hati and yi != pos_label:
            tn += 1
        elif yi != y_hati and yi == pos_label:
            fn += 1
        elif yi != y_hati and yi != pos_label:
            fp += 1
    return (tp, tn, fp, fn)


def accuracy_score_(
    y: np.ndarray, y_hat: np.ndarray, pos_label: Union[int, str] = 1, eps: float = 1e-15
) -> Optional[float]:
    """
    Compute the accuracy score.
    Args:
        y:a numpy.array for the correct labels
        y_hat:a numpy.array for the predicted labels
    Return:
        The accuracy score as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    if (
        not isinstance(y, np.ndarray)
        or not isinstance(y_hat, np.ndarray)
        or (not isinstance(pos_label, int) and not isinstance(pos_label, str))
        or y.size == 0
        or y.shape != y_hat.shape
    ):
        return None
    tp, tn, fp, fn = count_tp_tn_fp_fn(y, y_hat, pos_label)
    return (tp + tn) / (tp + fp + tn + fn + eps)


def precision_score_(
    y: np.ndarray, y_hat: np.ndarray, pos_label: Union[int, str] = 1, eps: float = 1e-15
) -> Optional[float]:
    """
    Compute the precision score.
    Args:
        y:a numpy.array for the correct labels
        y_hat:a numpy.array for the predicted labels
        pos_label: str or int, the class on which to report the precision_score (default=1)
    Return:
        The precision score as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    if (
        not isinstance(y, np.ndarray)
        or not isinstance(y_hat, np.ndarray)
        or (not isinstance(pos_label, int) and not isinstance(pos_label, str))
        or y.size == 0
        or y.shape != y_hat.shape
    ):
        return None
    tp, tn, fp, fn = count_tp_tn_fp_fn(y, y_hat, pos_label)
    return tp / (tp + fp + eps)


def recall_score_(
    y: np.ndarray, y_hat: np.ndarray, pos_label: Union[int, str] = 1, eps: float = 1e-15
) -> Optional[float]:
    """
    Compute the recall score.
    Args:
        y:a numpy.array for the correct labels
        y_hat:a numpy.array for the predicted labels
        pos_label: str or int, the class on which to report the precision_score (default=1)
    Return:
        The recall score as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    if (
        not isinstance(y, np.ndarray)
        or not isinstance(y_hat, np.ndarray)
        or (not isinstance(pos_label, int) and not isinstance(pos_label, str))
        or y.size == 0
        or y.shape != y_hat.shape
    ):
        return None
    tp, tn, fp, fn = count_tp_tn_fp_fn(y, y_hat, pos_label)
    return tp / (tp + fn + eps)


def f1_score_(
    y: np.ndarray, y_hat: np.ndarray, pos_label: Union[int, str] = 1, eps: float = 1e-15
) -> Optional[float]:
    """
    Compute the f1 score.
    Args:
        y:a numpy.array for the correct labels
        y_hat:a numpy.array for the predicted labels
        pos_label: str or int, the class on which to report the precision_score (default=1)
    Return:
        The f1 score as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    if (
        not isinstance(y, np.ndarray)
        or not isinstance(y_hat, np.ndarray)
        or (not isinstance(pos_label, int) and not isinstance(pos_label, str))
        or y.size == 0
        or y.shape != y_hat.shape
    ):
        return None
    precision = precision_score_(y, y_hat, pos_label)
    recall = recall_score_(y, y_hat, pos_label)
    return (2 * precision * recall) / (precision + recall + eps)
