import numpy as np
from cvar import *
from asymp_var import *
import multiprocessing as mp
from burr import Burr
from frechet import Frechet

def mse(x, true):
    return np.nanmean((x-true)**2, axis=0)

if __name__ == '__main__':
    np.random.seed(7)
    alph = 0.999
    D = Burr(0.75, 2)
    s = 2000
    n = 100000
    k = np.ceil(n**(2/3)).astype(int)
    t = n/k
    beta = k/(n*(1-alph))

    cvar_true = D.cvar(alph)

    data = D.rand((s, n))

    n_cpus = mp.cpu_count()
    pool = mp.Pool(n_cpus)

    result = [pool.apply_async(cvar_pot,
                args=(x, alph, k, D)) for x in data]

    cvar_pot_est = []
    for r in result:
        cvar_pot_est.append(r.get())
    cvar_pot_est = np.asarray(cvar_pot_est)

    c = cvar_pot_est[:,0]
    xi = cvar_pot_est[:,1]
    sig = cvar_pot_est[:,2]

    approx_error = [D.cvar_approx_error(t, alph, p1, p2) for p1,p2 in zip(xi,sig)]
    approx_error = np.asarray(approx_error)

    cvar_unbiased = c - approx_error

    avar_est = sig**2 * np.array([asymp_var(x, beta)/k for x in xi])
    avar_true = D.a(t)**2 * asymp_var(D.xi, beta)/k
    MSE = mse(cvar_unbiased, cvar_true)

    xi_eff = (1+D.xi)**2/k/mse(xi,D.xi)
    sig_eff = (D.a(t)**2*(1+(1+D.xi)**2))/k/mse(sig,D.a(t))

    print("Xi Efficiency: {:.3f}".format(xi_eff))
    print("Sigma Efficiency: {:.3f}".format(sig_eff))

    print("True CVaR: {:.3f}".format(cvar_true))
    print("POT CVaR (mean): {:.3f}".format(np.mean(c)))
    print("Unbiased CVaR (mean): {:.3f}".format(np.mean(cvar_unbiased)))
    print('')
    print("True avar: {:.3f}".format(avar_true))
    print("Estimated avar (mean): {:.3f}".format(np.mean(avar_est)))
    print("Mean squared error: {:.3f}".format(MSE))
    print("Efficiency: {:.3f}".format(np.mean(avar_est)/MSE))
