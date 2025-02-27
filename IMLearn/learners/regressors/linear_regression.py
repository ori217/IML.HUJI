from __future__ import annotations
from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import pinv
from sklearn.metrics import mean_squared_error as mse
from ...metrics import mean_square_error
import pandas as pd

class LinearRegression(BaseEstimator):
    """
    Linear Regression Estimator

    Solving Ordinary Least Squares optimization problem
    """

    def __init__(self, include_intercept: bool = True) -> LinearRegression:
        """
        Instantiate a linear regression estimator

        Parameters
        ----------
        include_intercept: bool, default=True
            Should fitted model include an intercept or not

        Attributes
        ----------
        include_intercept_: bool
            Should fitted model include an intercept or not

        coefs_: ndarray of shape (n_features,) or (n_features+1,)
            Coefficients vector fitted by linear regression. To be set in
            `LinearRegression.fit` function.
        """
        super().__init__()
        self.include_intercept_, self.coefs_ = include_intercept, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit Least Squares model to given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----
        Fits model with or without an intercept depending on value of `self.include_intercept_`
        """
        # computes the best func (meaning coef vec w-hat)
        # by finding one with the least squared error.
        # compute it with pseudo inverse:
        if self.include_intercept_:
            w0 = np.array([[1] for i in range(X.shape[0])])
            X = np.append(w0, X, 1)
        self.coefs_ = np.linalg.pinv(X).dot(y)



    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        if self.include_intercept_:
            w0 = np.array([[1] for i in range(X.shape[0])])
            X = np.append(w0, X, 1)
        return X.dot(self.coefs_)


    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under MSE loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under MSE loss function
        """
        # under the assumption that using mse func from sklearn is fine
        # according to Q in https://moodle2.cs.huji.ac.il/nu21/mod/moodleoverflow/discussion.php?d=1356
        pred = self._predict(X)
        k = mean_square_error(y, pred)
        return k


