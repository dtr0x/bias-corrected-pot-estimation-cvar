import numpy as np
from scipy.stats import t as student
from distribution import Distribution

class HalfT(Distribution):
    def __init__(self, df):
        self.df = df
        self.xi = 1/df
        self.rho = -2/df

    def get_label(self):
        return "half-t({})".format(round(self.df, 2))

    def cdf(self, x):
        return 2*student.cdf(x, self.df)

    def pdf(self, x):
        return 2*student.pdf(x, self.df)

    def var(self, alph):
        return student.ppf((alph+1)/2, self.df)

    def cvar(self, alph):
        df = self.df
        q = self.var(alph)
        return (df + q**2)/(df-1)/(1-alph) * self.pdf(q)
