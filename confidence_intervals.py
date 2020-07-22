import numpy as np
from scipy.stats import norm
from gen_data import gen_data
from burr import Burr
from frechet import Frechet

if __name__ == '__main__':
    # parameters for simulations
    dists = [Burr(0.5, 3), Burr(0.75, 2), Frechet(1.2), Frechet(1.6)]
    labels = ['Burr(0.5, 3)', 'Burr(0.75, 2)', 'Fr\\\'echet(1.2)', 'Fr\\\'echet(1.6)']
    n_samples = 5000
    sampsizes = np.array([1000, 2000, 5000, 10000, 25000, 50000, 100000])
    n_excesses = np.ceil(sampsizes**(2/3)).astype(int)
    alph = 0.999
    delt = 0.1

    if not all([v in globals() for v in ['cvar_pot_est', 'xi_est', 'sig_est', \
                                            'approx_error', 'avar']]):
        cvar_pot_est, xi_est, sig_est, approx_error, avar = \
        gen_data(dists, alph, n_samples, sampsizes, n_excesses)

    cvar_true = [d.cvar(alph) for d in dists]

    cvar_unbiased = cvar_pot_est - approx_error
    asymp_stderr = sig_est * np.sqrt(avar)
    ci_lower = cvar_unbiased - asymp_stderr*norm.ppf(1-delt/2)
    ci_upper = cvar_unbiased + asymp_stderr*norm.ppf(1-delt/2)

    rmse = []
    for i in range(len(cvar_true)):
        mse = np.nanmean((cvar_unbiased[i]-cvar_true[i])**2, axis=0)
        rmse.append(np.sqrt(mse))
    rmse = np.asarray(rmse)

    cov_prob = []
    nan_rate = []
    for c, l, u, lab in zip(cvar_true, ci_lower, ci_upper, labels):
        nan_rate.append(np.isnan(l).sum()/l.size * 100)
        coverage = []
        for i in range(l.shape[1]):
            lw = l[:, i]
            up = u[:, i]
            lw = lw[~np.isnan(lw)]
            up = up[~np.isnan(up)]
            ci = np.array([lw, up]).transpose()
            cvrg = sum([c > intv[0] and c < intv[1] for intv in ci])/ci.shape[0]
            coverage.append(cvrg)
        cov_prob.append(coverage)
    cov_prob = np.asarray(cov_prob)

    cvar_avg = np.nanmean(cvar_unbiased, axis=1)
    ae_avg = np.nanmean(approx_error, axis=1)
    avar_avg = np.nanmean(avar, axis=1)

    np.savez_compressed('data/results.npz', cov_prob=cov_prob, \
        cvar_est=cvar_avg, approx_error=ae_avg, avar=avar_avg, \
        dist_labels=labels, dists=dists, cvar_true=cvar_true, nan_rate=nan_rate, \
        n_excesses=n_excesses, delt=delt, alph=alph, rmse=rmse)
