import numpy as np

class Distribution:
    def rand(self, n):
        p = np.random.uniform(size=n)
        return self.var(p)

    def var_approx(self, t, alph):
        xi = self.xi
        sig = self.a(t)
        u = self.var(1-1/t)
        return u + sig/xi * ((t*(1-alph))**(-xi) - 1)

    def cvar_approx(self, t, alph):
        xi = self.xi
        q = self.var_approx(t, alph)
        sig = self.a(t)
        u = self.var(1-1/t)
        return (q + sig - xi*u)/(1-xi)

    def I(self, t, alph):
        xi = self.xi
        rho = self.rho
        s = 1/t/(1-alph)
        return 1/rho * ((s**(xi+rho)-1)/(xi+rho) - (s**xi-1)/xi)

    def var_approx_error(self, t, alph):
        sig = self.a(t)
        A = self.A(t)
        I = self.I(t, alph)
        return -sig * A * I

    def cvar_approx_error(self, t, alph, xi=None, sig=None):
        A = self.A(t)
        if A == 0: # no approximation error
            return 0
        rho = self.rho
        if not (xi and sig):
            xi = self.xi
            sig = self.a(t)
        s = 1/t/(1-alph)
        x1 = s**xi/xi/(1-xi)
        x2 = s**(xi+rho)/(xi+rho)/(1-xi-rho)
        x3 = rho/xi/(xi+rho)
        return sig * A * (x1 - x2 - x3) / rho

    def mle_bias(self, xi_mle, n, k):
        A = self.A(n/k)
        if A == 0: # no bias
            return (0, 0, 0)
        else:
            rho = self.rho
            g = 1-rho
            a = -g*A
            b = g**2 + g*xi_mle + A
            c = -xi_mle - 1
            b_xi = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
            b_sig = -rho/((1-rho)*(1+xi_mle-A*b_xi-rho))
            return b_xi, b_sig, A

    def params_est(self, xi_mle, sig_mle, n, k):
        b_xi, b_sig, A = self.mle_bias(xi_mle, n, k)
        return xi_mle-A*b_xi, sig_mle/(1+A*b_sig)
