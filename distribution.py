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

    def approx_error(self, t, alph):
        xi = self.xi
        rho = self.rho
        a = self.a(t)
        A = self.A(t)
        s = 1/t/(1-alph)
        if xi + rho != 0:
            x = s**xi/xi/(1-xi)
            y = s**(xi+rho)/(1-xi-rho) + rho/xi
            z = 1/(xi+rho)
            return a*A*(x-z*y)/rho
        else:
            x = s**xi/xi/(1-xi)
            y = np.log(s)
            z = (xi-1)/xi
            return a*A*(x-y+z)/rho

    def U(self, t):
        return self.var(1-1/t)
