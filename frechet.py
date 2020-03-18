import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma as G, gammaincc as Ginc

class Frechet():
    def cdf(self, x, gamma):
        return np.exp(-x**(-gamma))

    def rand(self, n, gamma):
        p = np.random.uniform(size=n)
        return self.var(p, gamma)

    def var(self, alph, gamma):
        return (-np.log(alph))**(-1/gamma)

    def cvar(self, alph, gamma):
        a = (gamma-1)/gamma
        x = -np.log(alph)
        return 1/(1-alph) * (G(a) - Ginc(a,x)*G(a))

    def var_approx(self, u, alph, gamma):
        xi = 1/gamma
        sig = self.sigma(u,gamma)
        Fbar = self.tau(u,gamma)
        return u + sig/xi * (((1-alph)/Fbar)**(-xi) - 1)

    def cvar_approx(self, u, alph, gamma):
        q = self.var_approx(u, alph, gamma)
        xi = 1/gamma
        sig = self.sigma(u,gamma)
        return q/(1-xi) + (sig-xi*u)/(1-xi)

    def A(self, t, gamma):
        return -(1+gamma+t*gamma*np.log(1-1/t)) \
                /((t-1)*gamma*np.log(1-1/t)) - 1/gamma

    def tau(self, u, gamma):
        return 1 - self.cdf(u, gamma)

    def sigma(self, u, gamma):
        t = self.tau(u, gamma)
        return t*(-np.log(1-t))**(-1-1/gamma) / (1-t) / gamma

    def s(self, u, alph, gamma):
        return self.tau(u, gamma)/(1-alph)

    def I(self, u, alph, gamma):
        t = self.s(u,alph,gamma)
        rho = -1
        xi = 1/gamma
        return 1/rho * (t**(xi+rho)/(xi+rho) - t**xi/xi) + 1/xi/(xi+rho)

    def var_bounds(self, u, eta, alph, gamma):
        t = self.s(u,alph,gamma)
        rho = -1
        xi = 1/gamma
        sig = self.sigma(u,gamma)
        A_tau = self.A(1/self.tau(u,gamma),gamma)
        I_t = self.I(u, alph, c, k)
        L = sig * (1-eta) * t**(-eta) * A_tau * I_t
        U = sig * (1+eta) * t**eta * A_tau * I_t
        if L > U:
            tmp = U
            U = L
            L = tmp
        return L, U

    def cvar_bounds(self, u, eta, alph, gamma):
        t = self.s(u,alph,gamma)
        rho = -1
        xi = 1/gamma
        L = self.sigma(u,gamma) * (1-eta) * t**(-eta) \
            * self.A(1/self.tau(u,gamma),gamma)
        U = self.sigma(u,gamma) * (1+eta) * t**eta \
            * self.A(1/self.tau(u,gamma),gamma)
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
