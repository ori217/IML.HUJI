from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from math import floor
from statistics import mean, stdev
pio.templates.default = "simple_white"

DATE_COL = ["date"]
NO_NEG_COL = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
       'waterfront', 'view', 'condition', 'grade', 'sqft_above',
       'sqft_basement', 'yr_built', 'yr_renovated',
       'sqft_living15', 'sqft_lot15']


def remove_zeros(field: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    clean field column from unneccesary zeros
    field: column name to work on
    df: the DataFrame
    _________
    return: fixed DF
    """
    # df = df.fillna(0) - dont need as we have dropna
    df[field] = df[field].replace("0",0)
    return df.drop(df[df[field] == 0.0].index)

def remove_neg(field: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    removes rows with negative values for field
    field: column name to work on
    df: the DataFrame
    _________
    return: fixed DF
    """
    return df.drop(df[df[field] < 0].index)


def change_2_datetime(field: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    for given df and field, changes field to DF. field needs to be clean
    # TODO: add format?
    field: column name to work on
    df: the DataFrame
    _________
    return: fixed DF
    """
    df[field] = df[field].apply(pd.to_datetime)
    return df


def load_data(filename: str) -> (pd.DataFrame,pd.Series):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename).dropna().drop_duplicates()

    #data changes

    #fixes
    remove_zeros("zipcode",df)
    for col in DATE_COL:
        df = change_2_datetime(col,remove_zeros(col,df))
    for col in NO_NEG_COL:
        df = remove_neg(col,df)

    #gets dummies from zipcode
    df = df.join(pd.get_dummies(df.zipcode))

    # new fields
    df["year"] = df.date.dt.year
    df["month"] = df.date.dt.month

    response_v = df.price
    domain = df.drop(["price", "date", "zipcode", "lat", "long", "id"], axis=1)

    return domain,response_v


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    for col in X.columns:
        pear = X[col].cov(y) /(X[col].std() * y.std())
        fig = go.Figure()
        fig.add_scatter(x=X[col], y=y, mode="markers")
        fig.update_layout(title=f"Evaluation of {col} with cor {round(pear, 3)}")
        fig.write_image(f"{output_path}/{col}.png")
        # pio.show(fig)




if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    filepath = r"/Users/or/Documents/uni/semester B/IML/repo/IML.HUJI/datasets/house_prices.csv"
    domain, response_v = load_data(filepath)



    # Question 2 - Feature evaluation with respect to response
    output_path = r"/Users/or/Documents/uni/semester B/IML/exes/exe2"
    feature_evaluation(domain,response_v,output_path)

    # Question 3 - Split samples into training- and testing sets.

    train_X, train_y, test_X, test_y = split_train_test(domain, response_v)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set

    m = train_X.shape[0] + 1
    means = []
    stds = []
    for p in range(10, 101):
        losses = []
        for i in range(10):
            cur_train_X = train_X.sample(frac=(p/100))
            cur_train_y = train_y.loc[cur_train_X.index]
            est = LinearRegression()
            est.fit(np.array(cur_train_X), np.array(cur_train_y))
            losses.append(est.loss(np.array(test_X), np.array(test_y)))
        losses = np.array(losses)
        means.append(losses.mean())
        stds.append(losses.std())

    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    x = list(range(10, 101))
    loss_means = np.array(means)
    loss_stds = np.array(stds)
    plot1 = go.Scatter(x=x, y=loss_means, mode="markers+lines",
                       name="Mean Loss", line=dict(dash="dash"), marker=dict(color="blue", opacity=.8))
    plot2 = go.Scatter(x=x, y=loss_means - 2 * loss_stds, fill=None, mode="lines",
                       line=dict(color="lightgrey"), showlegend=False)
    plot3 = go.Scatter(x=x, y=loss_means + 2 * loss_stds, fill='tonexty', mode="lines",
                       line=dict(color="lightgrey"), showlegend=False)
    fig = go.Figure()
    fig.add_trace(plot1)
    fig.add_trace(plot2)
    fig.add_trace(plot3)
    fig.update_layout(title_text="Mean loss with growing data size",
                      xaxis_title="% of train data", yaxis_title="Mean Loss")
    fig.show()



