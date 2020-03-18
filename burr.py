import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hyp2f1
from scipy.stats import burr12

class Burr():
    def cdf(self, x, c, k):
        return 1 - (1 + x**c)**(-k)

    def rand(self, n, c, k):
        return burr12.rvs(c, k, size=n)

    def var(self, alph, c, k):
        return ((1-alph)**(-1/k) - 1)**(1/c)

    def cvar(self, alph, c, k):
        Fv = 1-alph
        r = -1/c
        s = k-1/c
        t = 1-1/c+k
        return c*k/(c*k-1) * Fv**(-1/(c*k)) * hyp2f1(r, s, t, Fv**(1/k))

    def var_approx(self, u, alph, c, k):
        xi = 1/c/k
        sig = self.sigma(u,c,k)
        Fbar = self.tau(u,c,k)
        return u + sig/xi * (((1-alph)/Fbar)**(-xi) - 1)

    def cvar_approx(self, u, alph, c, k):
        q = self.var_approx(u, alph, c, k)
        xi = 1/c/k
        sig = self.sigma(u,c,k)
        return q/(1-xi) + (sig-xi*u)/(1-xi)

    def A(self, t, c, k):
        return (1-c)/(c * k * (t**(1/k) - 1))

    def tau(self, u, c, k):
        return 1 - self.cdf(u, c, k)

    def sigma(self, u, c, k):
        t = self.tau(u,c,k)
        return 1/c/k * t**(-1/k) * (t**(-1/k) - 1)**(1/c - 1)

    def s(self, u, alph, c, k):
        return self.tau(u, c, k)/(1-alph)

    def I(self, u, alph, c, k):
        t = self.s(u,alph,c,k)
        rho = -1/k
        xi = 1/c/k
        return 1/rho * (t**(xi+rho)/(xi+rho) - t**xi/xi) + 1/xi/(xi+rho)

    def var_bounds(self, u, eta, alph, c, k):
        t = self.s(u,alph,c,k)
        rho = -1/k
        xi = 1/c/k
        sig = self.sigma(u,c,k)
        A_tau = self.A(1/self.tau(u,c,k),c,k)
        I_t = self.I(u, alph, c, k)
        L = sig * (1-eta) * t**(-eta) * A_tau * I_t
        U = sig * (1+eta) * t**eta * A_tau * I_t
        if L > U:
            tmp = U
            U = L
            L = tmp
        return L, U

    def cvar_bounds(self, u, eta, alph, c, k):
        t = self.s(u,alph,c,k)
        rho = -1/k
        xi = 1/c/k
        L = self.sigma(u,c,k) * (1-eta) * t**(-eta) \
            * self.A(1/self.tau(u,c,k),c,k)
        U = self.sigma(u,c,k) * (1+eta) * t**eta \
            * self.A(1/self.tau(u,c,k),c,k)
        L = L * (\
        t**(xi+rho)/rho/(xi+rho)/(1+eta-xi-rho) \
        - t**xi/rho/xi/(1+eta-xi) \
        + 1/xi/(xi+rho)/(1+eta) \
        )
        U = U * (\
        t**(xi+rho)/rho/(xi+rho)/(1-eta-xi-rho) \
        - t**xi/rho/xi/(1-eta-xi) \
        + 1/xi/(xi+rho)/(1-eta) \
        )
        if L > U:
            tmp = U
            U = L
            L = tmp
        return L, U
