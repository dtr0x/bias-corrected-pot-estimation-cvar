import numpy as np
from scipy.special import gamma as G, gammaincc as Ginc
from distribution import Distribution

class Frechet(Distribution):
    def __init__(self, gamma):
        self.gamma = gamma
        self.xi = 1/gamma
        self.rho = -1
        xi = self.xi
        rho = self.rho

    def get_label(self):
        return "Fr\\\'echet({})".format(round(self.gamma, 2))

    def cdf(self, x):
        gamma = self.gamma
        return np.exp(-x**(-gamma))

    def pdf(self, x):
        gamma = self.gamma
        return gamma * x**(-1-gamma) * self.cdf(x)

    def a(self, t):
        gamma = self.gamma
        return 1/(gamma*(t-1)) * np.log(t/(t-1))**(-1/gamma-1)

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


    # moment of truncated Frechet distribution
    def mom(self, r, alph):
        q = self.var(alph)
        g = self.gamma
        k = 1-np.exp(-q**(-g))
        return (G(1-r/g) - Ginc(1-r/g, q**(-g)) * G(1-r/g)) / k


    # asymptotic variance of sample average estimator for Frechet distribution
    def avar(self, alph):
        m1 = self.mom(1, alph)
        m2 = self.mom(2, alph)
        return (m2 - m1**2)/(1-alph)
