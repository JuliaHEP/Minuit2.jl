import numpy as np
from iminuit import Minuit
from iminuit.cost import BinnedNLL
from jacobi import jacobi


rng = np.random.default_rng(1)
t_mu, t_sigma = 1., 0.1
x = rng.normal(t_mu, t_sigma, 1000)
nx, xe = np.histogram(x, bins=50, range=(.5, 1.5))


def model(x, mu, sigma):
    from math import erf
    z = (x - mu) / (np.sqrt(2) * sigma)
    return (1 + np.vectorize(erf)(z)) * 0.5

def mgrad(x, *args):
    return jacobi(lambda p: model(x, *p), args)[0].T

cost = BinnedNLL(nx, xe, model, grad=mgrad)
m = Minuit(cost, mu=0, sigma=1)
m.migrad()
