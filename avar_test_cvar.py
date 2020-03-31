import numpy as np
from eva import *
from asymp_var import *
import multiprocessing as mp
from burr import Burr
from frechet import Frechet
import sys
from scipy.stats import norm

if __name__ == '__main__':
    np.random.seed(7)
    alph = 0.999
    D = Burr(0.5, 3)
    s = 2000
    n = 50000
    n_excesses = np.ceil(n**(2/3)).astype(int)
    Fu = 1 - n_excesses/n
    u = D.var(Fu)
    xi = D.xi
    sig = D.sigma(u)
    parms_mle = np.array((xi, sig)) + D.mle_bias(u)
    cvar_true = D.cvar(alph)
    cvar_biased = D.cvar_approx_params(u, alph, parms_mle[0], parms_mle[1])
    data = D.rand((s, n))

    n_cpus = mp.cpu_count()
    pool = mp.Pool(n_cpus)

    result = [pool.apply_async(cvar_evt,
                args=(x, alph, n_excesses)) for x in data]

    cvars = []
    for r in result:
        cvars.append(r.get())
    cvars = np.asarray(cvars)
    cvar_avg = np.nanmean(cvars, axis=0)

    avar = asymp_var_biased(xi, sig, parms_mle[0], parms_mle[1], Fu, alph)
    m = mse(cvars, cvar_biased)
    b = bias(cvars, cvar_biased)
    crb = avar/n_excesses
    eff = crb/m

    print("True CVaR: {:.3f}.".format(cvar_true))
    print("Biased CVaR: {:.3f}.".format(cvar_biased))
    print("Mean CVaR: {:.3f}.".format(cvar_avg))
    print("Bias: {:.3f}.".format(b))
    print("MSE: {:.3f}.".format(m))
    print("CRB: {:.3f}.".format(crb))
    print("Efficiency: {:.3f}.".format(eff))

    thresholds = np.sort(data)[:, -n_excesses]
    sigs = D.sigma(thresholds)
    mle_means = [np.array((xi, sig)) + D.mle_bias(u) for \
                    sig, u in zip(sigs, thresholds)]
    cvar_mle = np.array([D.cvar_approx_params(u, alph, *params) for \
                    u, params in zip(thresholds, mle_means)])
    cvar_evt = np.array([D.cvar_approx_params(u, alph, xi, sig) for \
                    sig, u in zip(sigs, thresholds)])
    b1 = cvar_biased - cvar_evt
    b2 = -D.cvar_bound(thresholds, alph)
    delt = 0.1
    conf_means = cvars - b1 - b2
    avars = np.array([asymp_var_biased(xi, sig, *parms, Fu, alph) for \
                sig, parms in zip(sigs, mle_means)])
    conf_std = np.sqrt(avars/n_excesses) * norm.ppf(1-delt/2)
    conf_ints = np.array((conf_means - conf_std, \
                conf_means + conf_std)).transpose()
    coverage = 0
    for c in conf_ints:
        if cvar_true >= c[0] and cvar_true <= c[1]:
            coverage += 1/s
    print("Coverage: {:.3f}.".format(coverage))
