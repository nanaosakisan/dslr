import numpy as np
from typing import Optional, List


class MinMaxNormalisation:
    def __init__(self, min_: float = None, max_: float = None) -> None:
        if min_ != None and max_ != None:
            self.min_ = min_
            self.max_ = max_

    def fit(self, X: np.ndarray) -> None:
        if not isinstance(X, np.ndarray) or X.size == 0:
            return None
        self.min_ = np.min(X)
        self.max_ = np.max(X)

    def transform(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Computes the normalized version of a non-empty numpy.array using the min-max standardization.
        Args:
            x: has to be an numpy.array, a vector.
        Return:
            x’ as a numpy.array.
            None if x is a non-empty numpy.array or not a numpy.array.
            None if x is not of the expected type.
        Raises:
            This function shouldn’t raise any Exception.
        """
        if not isinstance(X, np.ndarray) or X.size == 0:
            return None
        return (X - self.min_) / (self.max_ - self.min_)

    def fit_transform(self, X: np.ndarray) -> Optional[np.ndarray]:
        self.fit(X)
        return self.transform(X)

    def tranform_rev(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Revese the normalized version of a non-empty numpy.array using the min-max standardization.
        Args:
            x: has to be an numpy.array, a vector.
        Return:
            x’ as a numpy.array.
            None if x is a non-empty numpy.array or not a numpy.array.
            None if x is not of the expected type.
        Raises:
            This function shouldn’t raise any Exception.
        """
        if not isinstance(X, np.ndarray) or X.size == 0:
            return None
        return X * (self.max_ - self.min_) + self.min_
