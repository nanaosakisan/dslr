from __future__ import annotations
import random
import numpy as np
from typing import Optional, Dict, Any


class MyLogisticRegression:
    """
    Description:
    My personnal logistic regression to classify things.
    """

    supported_penalities = ["l2"]

    def __init__(
        self,
        thetas: np.ndarray,
        alpha: float = 0.001,
        max_iter: int = 1000,
        penalty: str = "l2",
        lambda_: float = 1.0,
    ) -> None:
        if (
            not isinstance(alpha, float)
            or not isinstance(max_iter, int)
            or not isinstance(thetas, np.ndarray)
            or not isinstance(penalty, str)
            or not isinstance(lambda_, float)
        ):
            return None
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = thetas
        self.penalty = penalty
        self.lambda_ = lambda_ if penalty in self.supported_penalities else 0.0

    def get_params_(self) -> Dict[str, Any]:
        return self.__dict__

    def set_params_(self, **params) -> MyLogisticRegression:
        for p, v in params.items():
            setattr(self, p, v)
        return self

    def loss_elem(
        self,
        y: np.ndarray,
        y_hat: np.ndarray,
        eps: float = 1e-15,
    ) -> Optional[np.ndarray]:
        if (
            not isinstance(y, np.ndarray)
            or not isinstance(y_hat, np.ndarray)
            or not isinstance(eps, float)
            or y.size == 0
            or y_hat.size == 0
            or y.shape[0] != y_hat.shape[0]
            or y.shape[1] != 1
            or y_hat.shape[1] != 1
            or eps == 0
        ):
            return None
        ones = np.ones(y.shape[0]).reshape((-1, 1))
        theta_prime = np.append(0, np.delete(self.thetas, 0, 0)).reshape(
            (self.thetas.shape[0], 1)
        )
        J_elem = []
        for i in range(y):
            J_elem.append(
                -1
                / y.shape[0]
                * (
                    y[i] * np.log(y_hat[i] + eps)
                    + (ones[i] - y[i]) * np.log(ones[i] - y_hat[i] + eps)
                    + self.lambda_ / (2 * y.shape[0]) * theta_prime.T.dot(theta_prime)
                )
            )
        return np.array(J_elem)

    def reg_log_loss_(
        self,
        y: np.ndarray,
        y_hat: np.ndarray,
        eps: float = 1e-15,
    ) -> Optional[float]:
        """Computes the regularized loss of a logistic regression model from two non-empty numpy.array,
        without any for loop. The two arrays must have the same shapes.
        Args:
            y: has to be an numpy.array, a vector of shape m * 1.
            y_hat: has to be an numpy.array, a vector of shape m * 1.
            theta: has to be a numpy.array, a vector of shape n * 1.
            lambda_: has to be a float.
            eps: has to be a float, epsilon (default=1e-15).
        Return:
            The regularized loss as a float.
            None if y, y_hat, or theta is empty numpy.array.
            None if y or y_hat have component ouside [0 ; 1]
            None if y and y_hat do not share the same shapes.
            None if y or y_hat is not of the expected type.
        Raises:
            This function should not raise any Exception.
        """
        if (
            not isinstance(y, np.ndarray)
            or not isinstance(y_hat, np.ndarray)
            or not isinstance(self.thetas, np.ndarray)
            or not isinstance(self.lambda_, float)
            or not isinstance(eps, float)
            or y.size == 0
            or y.shape[1] != 1
            or y.shape != y_hat.shape
            or self.thetas.size == 0
            or self.thetas.shape[1] != 1
        ):
            return None
        ones = np.ones(y.shape[0]).reshape((-1, 1))
        theta_prime = np.append(0, np.delete(self.thetas, 0, 0)).reshape(
            (self.thetas.shape[0], 1)
        )
        ret = -1 / y.shape[0] * (
            y.T.dot(np.log(y_hat + eps))
            + (ones - y).T.dot(np.log((ones - y_hat) + eps))
        ) + self.lambda_ / (2 * y.shape[0]) * theta_prime.T.dot(theta_prime)
        return ret.item()

    def predict_(self, x: np.ndarray) -> Optional[np.ndarray]:
        """Computes the vector of prediction y_hat from two non-empty numpy.array.
        Args:
            x: has to be an numpy.array, a vector of shape m * n.
            theta: has to be an numpy.array, a vector of shape (n + 1) * 1.
        Return:
            y_hat: a numpy.array of shape m * 1, when x and theta numpy arrays
            with expected and compatible shapes.
            None: otherwise.
        Raises:
            This function should not raise any Exception.
        """
        if (
            not isinstance(x, np.ndarray)
            or not isinstance(self.thetas, np.ndarray)
            or x.size == 0
            or self.thetas.size == 0
            or self.thetas.shape[0] != x.shape[1] + 1
            or self.thetas.shape[1] != 1
        ):
            return None
        X_prime = np.array(np.c_[np.ones(len(x)), x], dtype=np.float)
        return 1 / (1 + np.exp(-X_prime.dot(self.thetas)))

    def vec_reg_logistic_grad(
        self, x: np.ndarray, y: np.ndarray
    ) -> Optional[np.ndarray]:
        """Computes the regularized logistic gradient of three non-empty numpy.array,
        without any for-loop. The three arrays must have compatible shapes.
        Args:
            y: has to be a numpy.array, a vector of shape m * 1.
            x: has to be a numpy.array, a matrix of dimesion m * n.
            theta: has to be a numpy.array, a vector of shape (n + 1) * 1.
            lambda_: has to be a float.
        Return:
            A numpy.array, a vector of shape (n + 1) * 1, containing the results of the formula for all j.
            None if y, x, or theta are empty numpy.array.
            None if y, x or theta does not share compatibles shapes.
            None if y, x or theta or lambda_ is not of the expected type.
        Raises:
            This function should not raise any Exception.
        """
        if (
            not isinstance(y, np.ndarray)
            or not isinstance(x, np.ndarray)
            or not isinstance(self.thetas, np.ndarray)
            or not isinstance(self.lambda_, float)
            or y.size == 0
            or x.size == 0
            or self.thetas.size == 0
            or y.shape[1] != 1
            or y.shape[0] != x.shape[0]
            or x.shape[1] != self.thetas.shape[0] - 1
            or self.thetas.shape[1] != 1
        ):
            return None
        X_prime = np.c_[np.ones(len(x)), x]
        theta_prime = np.append(0, np.delete(self.thetas, 0, 0)).reshape(
            (self.thetas.shape[0], 1)
        )
        return (
            1
            / y.shape[0]
            * (X_prime.T.dot(self.predict_(x) - y) + self.lambda_ * theta_prime)
        )

    def vec_reg_sgd(
        self, x: np.ndarray, y: np.ndarray, k: int = 40
    ) -> Optional[np.ndarray]:
        """Computes the regularized logistic stochiastic gradient of three non-empty
        numpy.array, without any for-loop. The three arrays must have compatible shapes.
        Args:
            y: has to be a numpy.array, a vector of shape m * 1.
            x: has to be a numpy.array, a matrix of dimesion m * n.
            k: has to be an int, a number so select how many feature will be used to
            estimate the predict thetas/
            theta: has to be a numpy.array, a vector of shape (n + 1) * 1.
            lambda_: has to be a float.
        Return:
            A numpy.array, a vector of shape (n + 1) * 1, containing the results of the formula for all j.
            None if y, x, or theta are empty numpy.array.
            None if y, x or theta does not share compatibles shapes.
            None if y, x or theta or lambda_ is not of the expected type.
        Raises:
            This function should not raise any Exception.
        """
        if (
            not isinstance(y, np.ndarray)
            or not isinstance(x, np.ndarray)
            or not isinstance(self.thetas, np.ndarray)
            or not isinstance(self.lambda_, float)
            or y.size == 0
            or x.size == 0
            or self.thetas.size == 0
            or y.shape[1] != 1
            or y.shape[0] != x.shape[0]
            or x.shape[1] != self.thetas.shape[0] - 1
            or self.thetas.shape[1] != 1
        ):
            return None
        data_tmp = np.c_[x, y]
        id_elem_ = random.sample(range(data_tmp.shape[0]), 40)
        data_sample = data_tmp[id_elem_]
        x = data_sample[:, :-1]
        y = data_sample[:, -1].reshape(-1, 1)
        X_prime = np.c_[np.ones(len(x)), x]
        return np.sum((-2 / k * X_prime) * (y - X_prime.dot(self.thetas)))

    def fit_(self, x: np.ndarray, y: np.ndarray, stochiastic: bool = False) -> None:
        """Computes a gradient vector from three non-empty numpy.array, without any for-loop.
        The three arrays must have compatible shapes.
        Args:
            x: has to be an numpy.array, a matrix of shape m * n.
            y: has to be an numpy.array, a vector of shape m * 1.
            theta: has to be an numpy.array, a vector (n +1) * 1.
        Return:
            The gradient as a numpy.array, a vector of shape n * 1,
            containing the result of the formula for all j.
            None if x, y, or theta are empty numpy.array.
            None if x, y and theta do not have compatible shapes.
            None if y, x or theta is not of the expected type.
        Raises:
            This function should not raise any Exception.
        """
        if (
            not isinstance(x, np.ndarray)
            or not isinstance(y, np.ndarray)
            or not isinstance(self.thetas, np.ndarray)
            or x.size == 0
            or y.size == 0
            or self.thetas.size == 0
            or y.shape[1] != 1
            or y.shape[0] != x.shape[0]
            or self.thetas.shape[1] != 1
            or self.thetas.shape[0] != x.shape[1] + 1
        ):
            return None
        if stochiastic == False:
            for _ in range(self.max_iter):
                self.thetas = self.thetas - self.alpha * self.vec_reg_logistic_grad(
                    x, y
                )
        else:
            for _ in range(self.max_iter):
                self.thetas = self.thetas - self.alpha * self.vec_reg_sgd(x, y, 40)
