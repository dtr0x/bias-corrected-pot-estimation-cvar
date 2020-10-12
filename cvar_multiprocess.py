import numpy as np
from cvar import cvar_pot, cvar_sa

# define functions for processing CVaR estimates in parallel

# iterate sample average CVaR
def sample_iter_sa(x, alph, sampsizes):
    cvars = []
    for n in sampsizes:
        c = cvar_sa(x[:n], alph)
        cvars.append(c)
    return np.array(cvars)

# iterate POT-based CVaR
def sample_iter_pot(x, alph, sampsizes, k, debias=True, k_rho=None):
    cvars = []
    if debias:
        if k_rho is not None:
            for n, ki, kr in zip(sampsizes, k, k_rho):
                c = cvar_pot(x[:n], alph, ki, debias, kr)
                cvars.append(c)
    else:
        for n, ki in zip(sampsizes, k):
            c = cvar_pot(x[:n], alph, ki, debias)
            cvars.append(c)
    return np.array(cvars)
