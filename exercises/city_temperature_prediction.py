import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import datetime as dt
pio.templates.default = "simple_white"

def check_temp(df:pd.DataFrame, min_t = -25, max_t = 50, temp_column ="Temp") -> pd.DataFrame:
    """
    removes rows that their temp isn't between min and max
    """
    return df.drop(df[df.Temp < min_t].index).drop(df[df.Temp > max_t].index)

def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=["Date"])
    df = df.dropna().drop_duplicates()
    df = check_temp(df)
    df["DayOfYear"] = df.Date.dt.dayofyear

    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data(r"/Users/or/Documents/uni/semester B/IML/repo/IML.HUJI/datasets/City_Temperature.csv")


    # Question 2 - Exploring data for specific country
    israel = df[df["Country"] == "Israel"]
    year_str = israel["Year"].apply(lambda x:str(x))
    fig1 = px.scatter(israel, x="DayOfYear", y="Temp", color=year_str,
                      title="Temp in Israel")
    fig1.show()

    agg_t = israel[["Month","Temp"]].groupby("Month").Temp.agg('std')
    months = list(range(1, 13))
    agg_data_df = pd.DataFrame({"Month": months, "Temps std": agg_t})
    fig2 = px.bar(agg_data_df, x="Month", y="Temps std", title="Std of Temps in Israel")
    fig2.show()

    # Question 3 - Exploring differences between countries
    gb = df[["Country", "Month", "Temp"]].groupby(["Country", "Month"])
    by_mean = gb.Temp.agg("mean").reset_index()
    by_std = gb.Temp.agg("std").reset_index()
    by_mean["Temp std"] = by_std["Temp"]
    by_mean = by_mean.rename(columns={"Temp": "Temp Mean"})
    fig3 = px.line(by_mean, x="Month", y="Temp Mean", color="Country", error_y="Temp std",
                   title="Avg temp by month and by country with errors")
    fig3.show()

    # # Question 4 - Fitting model for different values of `k`
    losses = {}
    X_deg, y_deg = israel["DayOfYear"], israel["Temp"]
    train_X_deg, train_y_deg, test_X_deg, test_y_deg = split_train_test(X_deg,y_deg)
    k_range = range(1, 11)
    for k in k_range:
        est = PolynomialFitting(k)
        est.fit(np.array(train_X_deg), np.array(train_y_deg))
        loss_k = est.loss(np.array(test_X_deg), np.array(test_y_deg))
        losses[k] = round(loss_k, 2)

    for deg in losses:
        print(f"Loss for degree {deg} is: {losses[deg]}")
    losses_df = pd.DataFrame({"Loss Degree": list(k_range),
                      "Loss": list(losses.values())})
    fig4 = px.bar(losses_df, x="Loss Degree", y="Loss", title="Loss as a function of the degree of the fitted polynomial")
    fig4.show()


    # Question 5 - Evaluating fitted model on different countries
    k = 5
    func = PolynomialFitting(k)
    func.fit(np.array(X_deg), np.array(y_deg))
    losses_other = []
    for country in df.Country.unique():
        country_data = df[df.Country == country]
        X_other ,y_other = country_data["DayOfYear"], country_data["Temp"]
        losses_other.append(func.loss(np.array(X_other), np.array(y_other)))

    losses_other_df = pd.DataFrame({"Country": list(df.Country.unique()),
                       "Loss": losses_other})
    fig5 = px.bar(losses_other_df, x="Country", y="Loss",
                  title="Error of model by Country")
    fig5.show()


