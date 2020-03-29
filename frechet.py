import numpy as np
from scipy.special import gamma as G, gammaincc as Ginc
from distribution import Distribution

class Frechet(Distribution):
    def __init__(self, gamma):
        self.gamma = gamma
        self.xi = 1/gamma
        self.rho = -1

    def cdf(self, x):
        gamma = self.gamma
        return np.exp(-x**(-gamma))

    def sigma(self, u):
        gamma = self.gamma
        t = self.tau(u)
        return t*(-np.log(1-t))**(-1-1/gamma) / (1-t) / gamma

    def var(self, alph):
        gamma = self.gamma
        return (-np.log(alph))**(-1/gamma)

    def cvar(self, alph):
        gamma = self.gamma
        a = (gamma-1)/gamma
        x = -np.log(alph)
        return 1/(1-alph) * (G(a) - Ginc(a,x)*G(a))

    def A(self, t):
        gamma = self.gamma
        return -(1+gamma+t*gamma*np.log(1-1/t)) \
                /((t-1)*gamma*np.log(1-1/t)) - 1/gamma
