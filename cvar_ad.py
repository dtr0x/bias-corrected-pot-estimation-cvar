import numpy as np
from scipy.stats import genpareto
from scipy.interpolate import interp1d
import warnings
from rho_adaptive import ada_rho
from estimators import A_est, debias_params, approx_error_est
import pandas as pd

'''Functions to compute CVaR using automated order statistic selection.'''


ad_quantiles = np.loadtxt("ADQuantiles.csv", delimiter=",",
                          dtype=float, skiprows=1, usecols=range(1, 1000))
ad_pvals = np.round(np.linspace(0.999, 0.001, 1000), 3)  # col names
ad_shape = np.round(np.linspace(-0.5, 1, 151), 2)  # row names


def var_sa(x, alph):
    return np.sort(x)[int(np.floor(alph*len(x)))]


def cvar_sa(x, alph):
    q = var_sa(x, alph)
    y = x[x >= q]
    return np.mean(y)


def cvar_evt(alph, u, xi, sigma, tp):
    s = (1-tp)/(1-alph)
    if xi == 0:
        return u + sigma * (np.log(s) + 1)
    else:
        return u + sigma/(1-xi) * (1+(s**xi - 1)/xi)


def get_excesses(x, tp):
    thresh = var_sa(x, tp)
    excesses = x[x > thresh] - thresh
    return thresh, excesses


def gpd_fit(y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xi_mle, _, sig_mle = genpareto.fit(y, floc=0)
    return xi_mle, sig_mle


def gpd_ad(x, tp):
    u, y = get_excesses(x, tp)
    xi, sigma = gpd_fit(y)
    z = genpareto.cdf(y, xi, 0, sigma)
    z = np.sort(z)
    n = len(z)
    i = np.linspace(1, n, n)
    stat = -n - (1/n) * np.sum((2 * i - 1) * (np.log(z) + np.log1p(-z[::-1])))
    return u, stat, xi, sigma


def ad_pvalue(stat, xi):
    row = np.where(ad_shape == round(np.clip(xi, -0.5, 1), 2))[0].item()
    if stat > ad_quantiles[row, -1]:
        xdat = ad_quantiles[row, 950:999]
        ydat = -np.log(ad_pvals[950:999])
        lfit = np.polyfit(xdat, ydat, 1)
        m = lfit[0]
        b = lfit[1]
        p = np.exp(-(m*stat+b))
    else:
        bound_idx = min(np.where(stat < ad_quantiles[row, ])[0])
        bound = ad_pvals[bound_idx]
        if bound == 0.999:
            p = bound
        else:
            x1 = ad_quantiles[row, bound_idx-1]
            x2 = ad_quantiles[row, bound_idx]
            y1 = -np.log(ad_pvals[bound_idx-1])
            y2 = -np.log(ad_pvals[bound_idx])
            lfit = interp1d([x1, x2], [y1, y2])
            p = np.exp(-lfit(stat))
    return p


def forward_stop(pvals, signif):
    pvals_transformed = np.cumsum(-np.log(1-pvals))/np.arange(1,len(pvals)+1)
    kf = np.where(pvals_transformed <= signif)[0]
    if len(kf) == 0:
        stop = 0
    else:
        stop = max(kf) + 1
    if stop == pvals.size:
        stop -= 1
    return kf, stop


def ad_test(x, alph, rho, tp_start, tp_end, tp_num):
    n = len(x)
    tps = np.linspace(tp_start, tp_end, tp_num)
    tests = []
    for tp in tps:
        u, stat, xi_mle, sig_mle = gpd_ad(x, tp)
        pval = ad_pvalue(stat, xi_mle)
        bpot = cvar_evt(alph, u, xi_mle, sig_mle, tp)

        k = len(x[x > u])
        A = A_est(x, k, xi_mle, rho)
        xi, sig = debias_params(xi_mle, sig_mle, rho, A)

        ae = approx_error_est(xi, sig, rho, A, alph, n, k)
        upot = cvar_evt(alph, u, xi, sig, tp) - ae

        tests.append([u, tp, pval, xi_mle, sig_mle, xi, sig, A, k, ae, bpot, upot])

    tests = pd.DataFrame(tests, columns=['u', 'tp', 'pval', 'xi_mle', \
        'sig_mle', 'xi', 'sig', 'A', 'k', 'approx_error', 'bpot', 'upot'])

    return tests


def cvar_ad(x, alph, tp_start=0.79, tp_end=0.98, tp_num=20, signif=0.1,
    cutoff=0.9):
    rho = ada_rho(x)
    tests = ad_test(x, alph, rho, tp_start, tp_end, tp_num)

    tests_b = tests[tests['xi_mle'] <= cutoff]
    tests_u = tests[tests['xi'] <= cutoff]

    b_idx = forward_stop(tests_b['pval'], signif)[1]
    u_idx = forward_stop(tests_u['pval'], signif)[1]

    stop_b = tests_b.index[b_idx] if b_idx >= 0 else np.nan
    stop_u = tests_u.index[u_idx] if u_idx >= 0 else np.nan

    bpot = tests.loc[stop_b].bpot if not np.isnan(stop_b) else np.nan

    if not np.isnan(stop_u):
        row = tests.loc[stop_u]
        if row.upot > 0:
            upot = row.upot
        else:
            upot = np.nan
        xi = row.xi
        sigma = row.sig
        k = row.k
        ae = row.approx_error
    else:
        upot, xi, sigma, k, ae = [np.nan]*5

    # sample average CVaR
    sa = cvar_sa(x, alph)

    return sa, bpot, upot, xi, sigma, rho, k, ae
