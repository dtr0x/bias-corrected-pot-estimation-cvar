import numpy as np
from scipy.special import hyp2f1
from distribution import Distribution

class Burr(Distribution):
    def __init__(self, c, k):
        self.c = c
        self.k = k
        self.xi = 1/c/k
        self.rho = -1/k

    def cdf(self, x):
        c = self.c
        k = self.k
        return 1 - (1 + x**c)**(-k)

    def sigma(self, u):
        c = self.c
        k = self.k
        t = self.tau(u)
        return 1/c/k * t**(-1/k) * (t**(-1/k) - 1)**(1/c - 1)

    def var(self, alph):
        c = self.c
        k = self.k
        return ((1-alph)**(-1/k) - 1)**(1/c)

    def cvar(self, alph):
        c = self.c
        k = self.k
        Fv = 1-alph
        r = -1/c
        s = k-1/c
        t = 1-1/c+k
        return c*k/(c*k-1) * Fv**(-1/(c*k)) * hyp2f1(r, s, t, Fv**(1/k))

    def A(self, t):
        c = self.c
        k = self.k
        return (1-c)/(c * k * (t**(1/k) - 1))
