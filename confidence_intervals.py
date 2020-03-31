import numpy as np
from scipy.stats import norm
from asymp_var import *

def dist_data(dists, Fus, alph):
    threshs = [[d.var(Fu) for Fu in Fus] for d in dists]
    cvars = [d.cvar(alph) for d in dists]
    xis = [d.xi for d in dists]
    sigs = [[d.sigma(u) for u in threshs] for d, threshs in \
                zip(dists, threshs)]
    return threshs, cvars, xis, sigs

def cvar_mle_bias(dist, xi, sig, u, alph):
    mle_mean = np.array((xi, sig)) + dist.mle_bias(u)
    mle_cvar = dist.cvar_approx_params(u, alph, *mle_mean)
    evt_cvar = dist.cvar_approx_params(u, alph, xi, sig)
    return mle_cvar - evt_cvar

def avar(dist, xi, sig, Fu, u, alph):
    mle_mean = np.array((xi, sig)) + dist.mle_bias(u)
    return asymp_var_biased(xi, sig, *mle_mean, Fu, alph)

def confidence_intervals(dists, n_excesses, Fus, alph, delt, cvars_evt):
    n_samples = cvars_evt.shape[1]
    threshs, cvars, xis, sigs = dist_data(dists, Fus, alph)
    coverage_probs = []
    confidence_intervals = []
    for i in range(len(dists)):
        d = dists[i]
        cvar_true = cvars[i]
        xi = xis[i]
        coverages = []
        conf_ints = []
        for j in range(len(n_excesses)):
            Fu = Fus[j]
            u = threshs[i][j]
            sig = sigs[i][j]

            bias1 = cvar_mle_bias(d, xi, sig, u, alph)
            bias2 = -d.cvar_bound(u, alph)

            cvars_corrected = cvars_evt[i, :, j] - bias1 - bias2

            av = avar(d, xi, sig, Fu, u, alph)/n_excesses[j]
            stdev = np.sqrt(av) * norm.ppf(1-delt/2)

            intervals = np.array((cvars_corrected - stdev, \
                        cvars_corrected + stdev)).transpose()
            coverage = 0
            for ci in intervals:
                if cvar_true >= ci[0] and cvar_true <= ci[1]:
                    coverage += 1/n_samples
            coverages.append(coverage)
            conf_ints.append(np.nanmean(intervals, axis=0))
        coverage_probs.append(coverages)
        confidence_intervals.append(conf_ints)
    return np.array(confidence_intervals), np.array(coverage_probs)
