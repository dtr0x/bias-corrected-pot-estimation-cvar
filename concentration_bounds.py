import numpy as np
from frechet import Frechet
from burr import Burr
from cvar import *
import matplotlib.pyplot as plt

def concentration_bound(n, alph, delt, b):
    # concentration bound for truncated estimator
    return 6*np.exp(-n*(1-alph)*delt**2/(48*b**2))

def trunc_thresh(p, B, delt, alph):
    beta = 1-alph
    return max((8*B/(beta*delt))**(1/(p-1)), (B/beta)**(1/p))

if __name__ == '__main__':
    np.random.seed(0)
    n = 10000000
    alph = 0.999
    #gamma = 4
    c = 3
    d = 4
    p = np.linspace(max(1.1, c*d/2), c*d, 1000)[:-1]
    #D = Frechet(gamma)
    D = Burr(c,d)
    B = D.moment(p) + 1 #np.ceil(F.moment(p))
    cvar_true = D.cvar(alph)
    delt = 0.1 * cvar_true
    x = D.rand(n)
    b = np.array([trunc_thresh(p_i, B_i, delt, alph) for p_i, B_i in zip(p, B)])
    c_bounds = concentration_bound(n, alph, delt, b)

    plt.plot(p, c_bounds)
    plt.show()
    plt.clf()
