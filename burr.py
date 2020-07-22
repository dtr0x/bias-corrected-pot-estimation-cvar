import numpy as np
from scipy.special import hyp2f1
from distribution import Distribution

class Burr(Distribution):
    def __init__(self, c, d):
        self.c = c
        self.d = d
        self.xi = 1/c/d
        self.rho = -1/d
        xi = self.xi
        rho = self.rho
        self.b = 1/(1-rho)/(1+xi-rho) * np.array((xi+1, -rho))

    def cdf(self, x):
        c = self.c
        d = self.d
        return 1 - (1 + x**c)**(-d)

    def a(self, t):
        c = self.c
        d = self.d
        return t**(1/d)/c/d * (t**(1/d) - 1)**(1/c - 1)

    def var(self, alph):
        c = self.c
        d = self.d
        return ((1-alph)**(-1/d) - 1)**(1/c)

    def cvar(self, alph):
        c = self.c
        d = self.d
        q = self.var(alph)
        return 1/(1-alph) * (d * ((1/q)**c)**(d-1/c))/(d-1/c) * \
                hyp2f1(d-1/c, 1+d, d-1/c+1, -1/q**c)

    def A(self, t):
        c = self.c
        d = self.d
        return (1-c)/(c * d * (t**(1/d) - 1))
