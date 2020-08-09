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
        self.b = 1/(1-rho)/(1+xi-rho) * np.array((xi+1, -rho))

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

    def moment(self, p):
        # requires p < gamma
        return G(1-p/self.gamma)
