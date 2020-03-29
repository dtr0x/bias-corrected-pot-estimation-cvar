import numpy as np
from scipy.special import hyp2f1
from scipy.stats import burr12, norm
from asymp_var import asymp_var

class Burr():
    def __init__(self, c, k):
        self.c = c
        self.k = k
        self.xi = 1/c/k
        self.rho = -1/k

    def cdf(self, x):
        c = self.c
        k = self.k
        return 1 - (1 + x**c)**(-k)

    def rand(self, n):
        c = self.c
        k = self.k
        return burr12.rvs(c, k, size=n)

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

    def var_approx(self, u, alph):
        xi = self.xi
        Fbar = self.tau(u)
        sig = self.aux_fun(1/Fbar)
        return u + sig/xi * (((1-alph)/Fbar)**(-xi) - 1)

    def cvar_approx(self, u, alph):
        xi = self.xi
        q = self.var_approx(u, alph)
        sig = self.aux_fun(1/self.tau(u))
        return q/(1-xi) + (sig-xi*u)/(1-xi)

    def var_approx_params(self, u, alph, xi, sig):
        Fbar = self.tau(u)
        return u + sig/xi * (((1-alph)/Fbar)**(-xi) - 1)

    def cvar_approx_params(self, u, alph, xi, sig):
        q = self.var_approx_params(u, alph, xi, sig)
        return q/(1-xi) + (sig-xi*u)/(1-xi)

    def A(self, t):
        c = self.c
        k = self.k
        return (1-c)/(c * k * (t**(1/k) - 1))

    def U(self, t):
        c = self.c
        k = self.k
        return (t**(1/k) - 1)**(1/c)

    def Uprime(self, t):
        c = self.c
        k = self.k
        return 1/c/k * t**(1/k-1) * (t**(1/k) - 1)**(1/c-1)

    def evtLim(self, x):
        xi = self.xi
        return (x**xi - 1)/xi

    def aux_fun(self, t):
        return t * self.Uprime(t)

    def tau(self, u):
        return 1 - self.cdf(u)

    def s(self, u, alph):
        return self.tau(u)/(1-alph)

    def I(self, u, alph):
        xi = self.xi
        rho = self.rho
        t = self.s(u,alph)
        if rho + xi != 0:
            return 1/rho * (t**(xi+rho)/(xi+rho) - t**xi/xi) + 1/xi/(xi+rho)
        else:
            return 1/xi * (t**xi/xi - 1/xi - np.log(t))

    def var_bound(self, u, alph):
        t = self.s(u,alph)
        r = 1/self.tau(u)
        sig = self.aux_fun(r)
        A_r = self.A(r)
        I_t = self.I(u, alph)
        return sig * A_r * I_t

    def cvar_bound(self, u, alph):
        xi = self.xi
        rho = self.rho
        t = self.s(u,alph)
        r = 1/self.tau(u)
        sig = self.aux_fun(r)
        A_r = self.A(r)
        x1 = t**(xi+rho)/rho/(xi+rho)/(1-xi-rho)
        x2 = t**xi/rho/xi/(1-xi)
        x3 = 1/xi/(xi+rho)
        return sig * A_r * (x1 - x2 + x3)

    def var_bounds(self, u, eta, alph):
        t = self.s(u,alph)
        sig = self.sigma(u)
        A_tau = self.A(1/self.tau(u))
        I_t = self.I(u, alph)
        L = sig * (1-eta) * t**(-eta) * A_tau * I_t
        U = sig * (1+eta) * t**eta * A_tau * I_t
        if L > U:
            tmp = U
            U = L
            L = tmp
        return L, U

    def cvar_bounds(self, u, eta, alph):
        xi = self.xi
        rho = self.rho
        t = self.s(u,alph)
        L = self.sigma(u) * (1-eta) * t**(-eta) \
            * self.A(1/self.tau(u))
        U = self.sigma(u) * (1+eta) * t**eta \
            * self.A(1/self.tau(u))
        if rho + xi != 0:
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
        else:
            L = L/xi * (t**xi/xi/(1+eta-xi) - np.log(t)/(1+eta) \
            -(1+eta+xi)/xi/(1+eta)**2)
            U = L/xi * (t**xi/xi/(1-eta-xi) - np.log(t)/(1-eta) \
            -(1-eta+xi)/xi/(1-eta)**2)
        if L > U:
            tmp = U
            U = L
            L = tmp
        return L, U

    def mle_bias(self, n, n_excesses):
        xi = self.xi
        rho = self.rho
        t = n/n_excesses
        sig = self.aux_fun(t)
        return self.A(t)/(1-rho)/(1+xi-rho) * np.array((xi+1, -sig*rho))
