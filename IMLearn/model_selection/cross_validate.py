from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    new_X = X
    new_y = y
    train_score = []
    validation_score = []
    m = len(y)
    for k in range(cv):
        train_x = np.concatenate((new_X[:int(m*(k/cv))], new_X[int(m*((k+1)/cv)):]))
        train_y = np.concatenate((new_y[:int(m*(k/cv))], new_y[int(m*((k+1)/cv)):]))
        test_x = new_X[int(m*(k/cv)):int(m*((k+1)/cv))]
        test_y = new_y[int(m*(k/cv)):int(m*((k+1)/cv))]

        try:
            estimator.fit(train_x.flatten(),train_y)
        except:
            estimator.fit(train_x, train_y)
        try:
            train_score.append(scoring(estimator.predict(train_x.flatten()),train_y))
        except:
            train_score.append(scoring(estimator.predict(train_x), train_y))
        try:
            validation_score.append(scoring(estimator.predict(test_x.flatten()),test_y))
        except:
            validation_score.append(scoring(estimator.predict(test_x), test_y))


    avg_train = sum(train_score) / len(train_score)
    avg_test = sum(validation_score) / len(validation_score)

    return avg_train, avg_test
