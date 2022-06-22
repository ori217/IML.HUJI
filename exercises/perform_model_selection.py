from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso
from IMLearn import BaseEstimator

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots

LOW_VAL = -1.2
HIGH_VAL = 2

def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions

    # scatter

    # create m=100 samples like that (noiseless)
    samples = list()
    for i in range(n_samples):
        # select X uniformly between -1.2 and 2
        x = np.random.uniform(LOW_VAL, HIGH_VAL)
        # generate gaussian noise eps with noise lvl 5
        y = (x+3)*(x+2)*(x+1)*(x-1)*(x-2)
        samples.append([x,y])

    samples = np.array(samples)
    X = samples[:,:-1]
    y = samples[:, -1]
    noisy_y = y + np.random.normal(0, noise, n_samples)

    # split dataset into train and test with prop of 2/3 with split_train_test (implemented)
    train_X, train_y, test_X, test_y = split_train_test(pd.DataFrame(X), pd.Series(noisy_y),2/3)
    train_X, train_y, test_X, test_y = np.array(train_X[0]), np.array(train_y), \
                                       np.array(test_X[0]), np.array(test_y)

    fig1 = go.Figure(layout=dict(title=dict(text=f'{n_samples} samples'
                                                 f' with noise {noise}')))
    fig1.add_trace(go.Scatter(x=X.flatten(), y=y.flatten(), mode='markers', name='noiseless'))
    fig1.add_trace(go.Scatter(x=train_X, y=train_y, mode='markers', name='train'))
    fig1.add_trace(go.Scatter(x=test_X, y=test_y, mode='markers', name='test'))
    fig1.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    train_X, test_X = np.expand_dims(train_X, axis=1), np.expand_dims(test_X, axis=1)
    train_y, test_y = np.expand_dims(train_y , axis = 1), np.expand_dims(test_y , axis = 1)
    train_err = []
    val_err = []
    degree = 10
    for k in range(degree):
        poly_fit = PolynomialFitting(k)
        avg_train, avg_test = cross_validate(poly_fit, train_X, train_y, mean_square_error)
        train_err.append(avg_train)
        val_err.append(avg_test)

    fig2 = go.Figure(layout=dict(title=dict(text='samples: ' + str(n_samples) + ' noise: ' + str(noise))))
    fig2.add_trace(go.Scatter(x=list(range(11)), y=train_err, mode='lines+markers', name='avg train_score'))
    fig2.add_trace(go.Scatter(x=list(range(11)), y=val_err, mode='lines+markers', name='avg test score'))
    # plot2.write_html('str(noise)' + '_noise_Q2.html', auto_open=False)
    fig2.show()


    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    k_star = 4 #err = 29.28, new err is 25.65
    poly_fit = PolynomialFitting(k_star)
    poly_fit.fit(train_X.flatten(), train_y)
    print(f"Loss is {round(poly_fit.loss(test_X.flatten(), test_y),2)}")

    # # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    # k_star = 10 #err = 0.2, new err is 0.0
    # poly_fit = PolynomialFitting(k_star)
    # poly_fit.fit(train_X.flatten(), train_y)
    # print(f"Loss is {round(poly_fit.loss(test_X.flatten(), test_y),2)}")



def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True, as_frame=True)
    train_X, train_y, test_X, test_y = split_train_test(X, y, n_samples/len(y))
    

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    ls = np.linspace(0.001, 4, n_evaluations)
    train_X, train_y, test_X, test_y = np.array(train_X), np.array(train_y), np.array(test_X), np.array(test_y)
    ridge_train_avgs = []
    ridge_test_avgs = []
    lasso_train_avgs = []
    lasso_test_avgs = []
    for lam in ls:
        ridge = RidgeRegression(lam)
        lasso = Lasso(alpha=lam)
        rid_avg_train, rid_avg_test = cross_validate(ridge, train_X, train_y, mean_square_error)
        las_avg_train, las_avg_test = cross_validate(lasso, train_X, train_y, mean_square_error)

        ridge_train_avgs.append(rid_avg_train), ridge_test_avgs.append(rid_avg_test)
        lasso_train_avgs.append(las_avg_train), lasso_test_avgs.append(las_avg_test)

    plot7 = go.Figure(layout=dict(title=dict(text='num of samples: ' + str(n_samples))))
    plot7.add_trace(go.Scatter(x=ls, y=ridge_train_avgs, mode='lines',
                               name='ridge avg train score'))
    plot7.add_trace(go.Scatter(x=ls, y=ridge_test_avgs, mode='lines',
                               name='ridge avg test score'))
    plot7.add_trace(go.Scatter(x=ls, y=lasso_train_avgs, mode='lines',
                               name='lasso avg train score'))
    plot7.add_trace(go.Scatter(x=ls, y=lasso_test_avgs, mode='lines',
                               name='lasso avg test score'))
    plot7.show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    best_ridge_lam = ls[ridge_test_avgs.index(min(ridge_test_avgs))]
    best_lasso_lam = ls[lasso_test_avgs.index(min(lasso_test_avgs))]
    print(f"Best reg values were {round(best_ridge_lam,2)} for ridge, and {round(best_lasso_lam,2)} for lasso")

    rid_est = RidgeRegression(best_ridge_lam).fit(train_X, train_y)
    lasso_est = Lasso(alpha=best_lasso_lam).fit(train_X, train_y)
    ls_est = LinearRegression().fit(train_X, train_y)
    for i in [rid_est, lasso_est, ls_est]:
        print(f'For estimator {i.__str__()} the score is: {mean_square_error(test_y, i.predict(test_X))}')





if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(100,noise=0)
    select_polynomial_degree(1500, noise = 10)
    select_regularization_parameter()



