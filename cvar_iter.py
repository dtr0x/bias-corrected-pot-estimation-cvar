import numpy as np
from cvar_ad import cvar_ad

# define function for processing CVaR estimates in parallel

def cvar_iter(x, alph, sampsizes):
    params = []
    for n in sampsizes:
        p = cvar_ad(x[:n], alph)
        params.append(p)
    return np.array(params)
