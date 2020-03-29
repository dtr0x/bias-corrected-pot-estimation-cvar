import numpy as np

class Distribution:
    def __init__(self, *params):
        pass

    def var(self, alph):
        pass

    def cvar(self, alph):
        pass

    def A(self, t):
        pass

    def sigma(self, u):
        pass

    def rand(self, n):
        p = np.random.uniform(size=n)
        return self.var(p)

    def var_approx(self, u, alph):
        xi = self.xi
        Fbar = self.tau(u)
        sig = self.sigma(u)
        return u + sig/xi * (((1-alph)/Fbar)**(-xi) - 1)

    def cvar_approx(self, u, alph):
        xi = self.xi
        q = self.var_approx(u, alph)
        sig = self.sigma(u)
        return q/(1-xi) + (sig-xi*u)/(1-xi)

    def var_approx_params(self, u, alph, xi, sig):
        Fbar = self.tau(u)
        return u + sig/xi * (((1-alph)/Fbar)**(-xi) - 1)

    def cvar_approx_params(self, u, alph, xi, sig):
        q = self.var_approx_params(u, alph, xi, sig)
        return q/(1-xi) + (sig-xi*u)/(1-xi)

    def tau(self, u):
        return 1 - self.cdf(u)

    def s(self, u, alph):
        return self.tau(u)/(1-alph)

    def I(self, u, alph):
        xi = self.xi
        rho = self.rho
        t = self.s(u,alph)
        return 1/rho * (t**(xi+rho)/(xi+rho) - t**xi/xi) + 1/xi/(xi+rho)

    def var_bound(self, u, alph):
        t = self.s(u,alph)
        sig = self.sigma(u)
        A_t = self.A(1/self.tau(u))
        I_t = self.I(u, alph)
        return sig * A_t * I_t

    def cvar_bound(self, u, alph):
        xi = self.xi
        rho = self.rho
        t = self.s(u,alph)
        sig = self.sigma(u)
        A_t = self.A(1/self.tau(u))
        x1 = t**(xi+rho)/rho/(xi+rho)/(1-xi-rho)
        x2 = t**xi/rho/xi/(1-xi)
        x3 = 1/xi/(xi+rho)
        return sig * A_t * (x1 - x2 + x3)

    def mle_bias(self, u):
        xi = self.xi
        rho = self.rho
        t = 1/self.tau(u)
        sig = self.sigma(u)
        return self.A(t)/(1-rho)/(1+xi-rho) * np.array((xi+1, -sig*rho))
