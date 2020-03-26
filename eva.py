import numpy as np
from scipy.stats import genpareto
import warnings

def get_excesses(x, n_excesses):
    excesses = np.sort(x)[-n_excesses:]
    thresh = excesses[0]
    return excesses - thresh

def gpdFit(x, n_excesses):
    y = get_excesses(x, n_excesses)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shape, _, scale = genpareto.fit(y, floc=0)
    return shape, scale

def rgpd(n, shape, scale):
    # Random number generation for the GPD
    return genpareto.rvs(shape, 0, scale, n)

def pgpd(x, shape, scale):
    # CDF for the GPD
    return genpareto.cdf(x, shape, 0, scale)

def qgpd(p, shape, scale):
    # Quantile function for the GPD
    return genpareto.ppf(p, shape, 0, scale)

def cvar_gpd(alph, shape, scale):
    q = qgpd(alph, shape, scale)
    return (q+scale)*(1+shape*q/scale)**(-1/shape)/((1-alph)*(1-shape))

def var_sa(x, alph):
    return np.sort(x)[int(np.ceil(alph*len(x)))]

def cvar_sa(x, alph):
	q = var_sa(x, alph)
	y = x[x >= q]
	return np.mean(y)

def var_evt(x, alph, n_excesses):
    u = np.sort(x)[-n_excesses]
    xi, sig = gpdFit(x, n_excesses)
    return u + sig/xi * (((1-alph)/(1-Fu))**(-xi) - 1)

def cvar_evt(x, alph, n_excesses):
    u = np.sort(x)[-n_excesses]
    xi, sig = gpdFit(x, n_excesses)
    if xi >= 1:
        return np.nan
    q = u + sig/xi * (((1-alph)/(1-Fu))**(-xi) - 1)
    return (q + sig - xi*u)/(1-xi)
