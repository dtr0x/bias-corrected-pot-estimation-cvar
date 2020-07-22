import numpy as np
from scipy.stats import genpareto
import warnings

def get_excesses(x, k):
    n = len(x)
    ord = np.sort(x)
    thresh = ord[n-k-1]
    excesses = ord[-k:] - thresh
    return thresh, excesses

def gpd_fit(y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xi_mle, _, sig_mle = genpareto.fit(y, floc=0)
    return xi_mle, sig_mle

def cvar_pot(x, alph, k, dist):
    u, y = get_excesses(x, k)
    n = len(x)
    k = len(y)
    Fu  = 1 - k/n
    xi_mle, sig_mle = gpd_fit(y)
    xi, sig = dist.params_est(xi_mle, sig_mle, n, k)
    beta = (1-alph)/(1-Fu)
    if xi >= 1:
        return np.nan, xi, sig
    else:
        q = u + sig/xi * (beta**(-xi) - 1)
        c = (q + sig - xi*u)/(1-xi)
        return c, xi, sig
