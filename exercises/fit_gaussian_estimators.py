from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"



def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    m, mu, sigma = 1000,10,1
    samples = np.random.normal(mu, sigma, m)
    unigaus = UnivariateGaussian()
    unigaus.fit(samples)
    print (unigaus.mu_,unigaus.var_)
    # print("fgsgfdsgdfg")

    # Question 2 - Empirically showing sample mean is consistent
    est_mean = []
    start , end , step, bins = 10,1001,10, 100
    ms = np.linspace(start , end , bins).astype(np.int)
    for s_size in range(start , end , step):
        sample = np.random.normal(mu, sigma, size=s_size)
        est_mean.append(sample.mean())
    print(len(ms))
    go.Figure([go.Scatter(x=ms, y=est_mean, mode='markers+lines', name=r'$\widehat\mu$'),
               go.Scatter(x=ms, y=[mu] * len(ms), mode='lines', name=r'$\mu$')],
              layout=go.Layout(title=r"$\text{(5) Estimation of Expectation As Function Of Number Of Samples}$",
                               xaxis_title="$m\\text{ - number of samples}$",
                               yaxis_title="Mean per Sample$",
                               height=500)).show()


    # Question 3 - Plotting Empirical PDF of fitted model
    go.Figure([go.Scatter(x=samples, y=unigaus.pdf(samples), mode='markers', name=r'$\widehat\var$'),
               # go.Scatter(x=ms, y=[mu] * len(ms), mode='lines', name=r'$\mu$')
               ],
              layout=go.Layout(title=r"$\text{(5) PDF As Function Of Number Of Samples}$",
                               xaxis_title="$m\\text{samples}$",
                               yaxis_title="calculated PDF$",
                               height=500)).show()





def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mul_size = 1000
    arr_mu = np.array([0, 0, 4, 0])
    arr_sigma = np.array(
                [[1, 0.2, 0, 0.5],
                [0.2, 2, 0, 0],
                [0, 0, 1, 0],
                [0.5, 0, 0, 1]])
    mul_var_s = np.random.multivariate_normal(arr_mu, arr_sigma, mul_size)
    mul_gaus = MultivariateGaussian()
    mul_gaus.fit(mul_var_s)

    print(mul_gaus.mu_)
    print(mul_gaus.cov_)


    # Question 5 - Likelihood evaluation
    lins = np.linspace(-10, 10, 200)
    arr = np.array([[0.0]*200]*200)
    for i, f1 in enumerate(lins):
        for j, f3 in enumerate(lins):
            mu = np.array([f1, 0, f3, 0])
            arr[i][j] = MultivariateGaussian.\
                log_likelihood(mu, arr_sigma, mul_var_s) ##

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(x=lins, y=lins, z=arr,
                   colorbar=dict(title="Log Likelihood")))
    fig.update_layout(
        title="Samples Log likelihood. More yellowish near true mu value",
        xaxis_title="f3",
        yaxis_title="f1")
    fig.show()


    # # Question 6 - Maximum likelihood
    max_model = np.argmax(arr)
    f1_val = lins[max_model//200]
    f3_val = lins[max_model % 200]
    print("Max-likelihood values of f1 is {} and of f3 is {}".format(f1_val,f3_val))

 
if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()

