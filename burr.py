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

    def aux_fun(self, n, k):
        return n/k * self.Uprime(n/k)

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
        sig = self.sigma(u)
        Fbar = self.tau(u)
        return u + sig/xi * (((1-alph)/Fbar)**(-xi) - 1)

    def cvar_approx(self, u, alph):
        xi = self.xi
        q = self.var_approx(u, alph)
        sig = self.sigma(u)
        return q/(1-xi) + (sig-xi*u)/(1-xi)

    def A2(self, t):
        c = self.c
        k = self.k
        return (1-c)/(c * k * (t**(1/k) - 1))

    def A(self, t):
        xi = self.xi
        rho = self.rho
        return xi * t**rho

    def tau(self, u):
        return 1 - self.cdf(u)

    def sigma(self, u):
        c = self.c
        k = self.k
        t = self.tau(u)
        return 1/c/k * t**(-1/k) * (t**(-1/k) - 1)**(1/c - 1)

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

    def sample_complexity(self, eps, delt, Fu, alph):
        xi = self.xi
        u = self.var(Fu)
        sig = self.sigma(u)
        psi = asymp_var(xi, sig, Fu, alph)
        norm_q = norm.ppf(1-delt/2)
        Bu = np.abs(self.cvar_bounds(u, 0, alph)[0])
        return psi/(1-Fu) * (norm_q/(eps-Bu))**2

    def mle_bias(self, n, k):
        #k = self.num_excesses(n)
        xi = self.xi
        rho = self.rho
        sig = self.aux_fun(n, k)
        return self.A2(n/k)/(1-rho)/(1+xi-rho) * np.array((xi+1, -sig*rho))

    def num_excesses(self, n):
        xi = self.xi
        rho = self.rho
        r = 2/(2+self.k)
        return int(n**r)
