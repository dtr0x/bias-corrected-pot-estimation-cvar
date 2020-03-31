import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from burr import Burr
from frechet import Frechet
from eva import *
from asymp_var import *
from scipy.stats import norm

def cvar_iter(x, alph, sampsizes, n_excesses):
    cvars = []
    for i in range(len(sampsizes)):
        cvars.append(cvar_evt(x[:sampsizes[i]], alph, n_excesses[i]))
    return np.asarray(cvars)

if __name__ == '__main__':
    np.random.seed(7)
    alph = 0.999
    delt = 0.1
    burr_params = ((3/4, 2), (1.5, 1))
    frec_params = (1.25, 2)
    Fu = 0.975
    s = 5000
    n = 100000
    n_excesses = [50, 100, 250, 500, 1000, 1750, 2500]
    sampsizes = [int(np.ceil(Nu/(1-Fu))) for Nu in n_excesses]

    # Burr all parameters
    burr_dists = [Burr(*p) for p in burr_params]
    burr_threshs = [b.var(Fu) for b in burr_dists]
    burr_cvars = [b.cvar(alph) for b in burr_dists]
    burr_xis = [b.xi for b in burr_dists]
    burr_sigs = [b.sigma(u) for b, u in zip(burr_dists, burr_threshs)]
    burr_mle_means = [np.array((xi, sig)) + b.mle_bias(u) for xi, sig, b, u in \
                        zip(burr_xis, burr_sigs, burr_dists, burr_threshs)]
    burr_mle_cvars = [b.cvar_approx_params(u, alph, *params) for b, u, params \
                        in zip(burr_dists, burr_threshs, burr_mle_means)]
    burr_bias_1 = [c - b.cvar_approx_params(u, alph, xi, sig) for c, b, u, \
                    xi, sig in zip(burr_mle_cvars, burr_dists, burr_threshs, \
                    burr_xis, burr_sigs)]
    burr_bias_2 = [-b.cvar_bound(u, alph) for b, u in \
                    zip(burr_dists, burr_threshs)]
    burr_avars = [asymp_var_biased(xi, sig, *params, Fu, alph) for \
                    xi, sig, params in \
                    zip(burr_xis, burr_sigs, burr_mle_means)]

    # Frechet all parameters
    frec_dists = [Frechet(p) for p in frec_params]
    frec_threshs = [f.var(Fu) for f in frec_dists]
    frec_cvars = [f.cvar(alph) for f in frec_dists]
    frec_xis = [f.xi for f in frec_dists]
    frec_sigs = [b.sigma(u) for b, u in zip(frec_dists, frec_threshs)]
    frec_mle_means = [np.array((xi, sig)) + b.mle_bias(u) for xi, sig, b, u in \
                        zip(frec_xis, frec_sigs, frec_dists, frec_threshs)]
    frec_mle_cvars = [b.cvar_approx_params(u, alph, *params) for b, u, params \
                        in zip(frec_dists, frec_threshs, frec_mle_means)]
    frec_bias_1 = [c - b.cvar_approx_params(u, alph, xi, sig) for c, b, u, \
                    xi, sig in zip(frec_mle_cvars, frec_dists, frec_threshs, \
                    frec_xis, frec_sigs)]
    frec_bias_2 = [-b.cvar_bound(u, alph) for b, u in \
                    zip(frec_dists, frec_threshs)]
    frec_avars = [asymp_var_biased(xi, sig, *params, Fu, alph) for \
                    xi, sig, params in \
                    zip(frec_xis, frec_sigs, frec_mle_means)]

    # generate random Burr variates
    try:
        if 'burr_data' not in locals():
            burr_data = np.load("data/burr_data.npy")
    except FileNotFoundError:
        burr_data = []
        for d in burr_dists:
            burr_data.append(d.rand((s,n)))
        burr_data = np.asarray(burr_data)
        np.save("data/burr_data.npy", burr_data)

    # generate random Frechet variates
    try:
        if 'frec_data' not in locals():
            frec_data = np.load("data/frec_data.npy")
    except FileNotFoundError:
        frec_data = []
        for d in frec_dists:
            frec_data.append(d.rand((s,n)))
        frec_data = np.asarray(frec_data)
        np.save("data/frec_data.npy", frec_data)

    # EVT CVaR data for Burr distributions
    try:
        if 'burr_cvars_evt' not in locals():
            burr_cvars_evt = np.load("data/burr_cvars_evt.npy")
    except FileNotFoundError:
        n_cpus = mp.cpu_count()
        pool = mp.Pool(n_cpus)
        burr_cvars_evt = []
        for data in burr_data:
            result = [pool.apply_async(cvar_iter,
                        args=(x, alph, sampsizes, n_excesses)) for x in data]
            cvars = []
            for r in result:
                cvars.append(r.get())
            burr_cvars_evt.append(np.asarray(cvars))
        burr_cvars_evt = np.asarray(burr_cvars_evt)
        np.save("data/burr_cvars_evt.npy", burr_cvars_evt)

    # EVT CVaR data for Frechet distributions
    try:
        if 'frec_cvars_evt' not in locals():
            frec_cvars_evt = np.load("data/frec_cvars_evt.npy")
    except FileNotFoundError:
        n_cpus = mp.cpu_count()
        pool = mp.Pool(n_cpus)
        frec_cvars_evt = []
        for data in frec_data:
            result = [pool.apply_async(cvar_iter,
                        args=(x, alph, sampsizes, n_excesses)) for x in data]
            cvars = []
            for r in result:
                cvars.append(r.get())
            frec_cvars_evt.append(np.asarray(cvars))
        frec_cvars_evt = np.asarray(frec_cvars_evt)
        np.save("data/frec_cvars_evt.npy", frec_cvars_evt)

    burr_coverage_probs = []
    burr_confidence_intervals = []
    for i in range(burr_cvars_evt.shape[0]):
        cvars_corrected = burr_cvars_evt[i] \
                                    + burr_bias_1[i] + burr_bias_2[i]
        avar = burr_avars[i]
        coverages = []
        conf_ints = []
        for j in range(len(n_excesses)):
            stdev = np.sqrt(avar/n_excesses[j]) * norm.ppf(1-delt/2)
            cvars = cvars_corrected[:, j]
            intervals = np.array((cvars - stdev, cvars + stdev)).transpose()
            coverage = 0
            for c in intervals:
                if burr_cvars[i] >= c[0] and burr_cvars[i] <= c[1]:
                    coverage += 1/s
            coverages.append(coverage)
            conf_ints.append(np.nanmean(intervals, axis=0))
        burr_coverage_probs.append(coverages)
        burr_confidence_intervals.append(conf_ints)
    burr_coverage_probs = np.asarray(burr_coverage_probs)
    burr_confidence_intervals = np.asarray(burr_confidence_intervals)

    frec_coverage_probs = []
    frec_confidence_intervals = []
    for i in range(frec_cvars_evt.shape[0]):
        cvars_corrected = frec_cvars_evt[i] \
                                    + frec_bias_1[i] + frec_bias_2[i]
        avar = frec_avars[i]
        coverages = []
        conf_ints = []
        for j in range(len(n_excesses)):
            stdev = np.sqrt(avar/n_excesses[j]) * norm.ppf(1-delt/2)
            cvars = cvars_corrected[:, j]
            intervals = np.array((cvars - stdev, cvars + stdev)).transpose()
            coverage = 0
            for c in intervals:
                if frec_cvars[i] >= c[0] and frec_cvars[i] <= c[1]:
                    coverage += 1/s
            coverages.append(coverage)
            conf_ints.append(np.nanmean(intervals, axis=0))
        frec_coverage_probs.append(coverages)
        frec_confidence_intervals.append(conf_ints)
    frec_coverage_probs = np.asarray(frec_coverage_probs)
    frec_confidence_intervals = np.asarray(frec_confidence_intervals)

    burr_cvar_means = np.nanmean(burr_cvars_evt, axis=1)
    for i in range(burr_cvar_means.shape[0]):
        p1, = plt.plot(n_excesses, burr_cvar_means[i], \
                linestyle='-', marker='.', color='k')
        p2, = plt.plot(n_excesses, burr_confidence_intervals[i, :, 0], \
                linestyle='--', marker='.', fillstyle='none', color='#383838')
        p3, = plt.plot(n_excesses, burr_confidence_intervals[i, :, 1], \
                linestyle='--', marker='.', fillstyle='none', color='#383838')
        p4, = plt.plot(n_excesses, [burr_cvars[i]]*len(n_excesses), \
                linestyle='-', color='#00208a')
        plt.xlabel("number of excesses")
        plt.ylabel("CVaR")
        plt.title("Burr{}, alpha={}".format(burr_params[i], alph))
        plt.legend([p1, p2, p4], \
            ["Mean CVaR", "Mean Confidence Intervals", "True CVaR"])
        plt.savefig("plots/burr/intervals_{}_{}_{}_{}.png".format(\
            burr_params[i][0], burr_params[i][1], alph, Fu),\
            bbox_inches="tight")
        plt.clf()

        plt.plot(n_excesses, burr_coverage_probs[i], \
                linestyle='--', color='k', marker='.')
        plt.plot(n_excesses, [1-delt]*len(n_excesses), linestyle='-', color='k')
        plt.xlabel("number of excesses")
        plt.ylabel("coverage probability")
        plt.title("Burr{}, alpha={}".format(burr_params[i], alph))
        plt.legend(labels=["Coverage", "Confidence Level"])
        plt.savefig("plots/burr/coverage_{}_{}_{}_{}.png".format(\
            burr_params[i][0], burr_params[i][1], alph, Fu),\
            bbox_inches="tight")
        plt.clf()

    frec_cvar_means = np.nanmean(frec_cvars_evt, axis=1)
    for i in range(frec_cvar_means.shape[0]):
        p1, = plt.plot(n_excesses, frec_cvar_means[i], \
                linestyle='-', marker='.', color='k')
        p2, = plt.plot(n_excesses, frec_confidence_intervals[i, :, 0], \
                linestyle='--', marker='.', fillstyle='none', color='#383838')
        p3, = plt.plot(n_excesses, frec_confidence_intervals[i, :, 1], \
                linestyle='--', marker='.', fillstyle='none', color='#383838')
        p4, = plt.plot(n_excesses, [frec_cvars[i]]*len(n_excesses), \
                linestyle='-', color='#00208a')
        plt.xlabel("number of excesses")
        plt.ylabel("CVaR")
        plt.title("Frechet({}), alpha={}".format(frec_params[i], alph))
        plt.legend([p1, p2, p4], \
            ["Mean CVaR", "Mean Confidence Intervals", "True CVaR"])
        plt.savefig("plots/frechet/intervals_{}_{}_{}.png".format(\
            frec_params[i], alph, Fu),\
            bbox_inches="tight")
        plt.clf()

        plt.plot(n_excesses, frec_coverage_probs[i], \
                linestyle='--', color='k', marker='.')
        plt.plot(n_excesses, [1-delt]*len(n_excesses), linestyle='-', color='k')
        plt.xlabel("number of excesses")
        plt.ylabel("coverage probability")
        plt.title("Frechet({}), alpha={}".format(frec_params[i], alph))
        plt.legend(labels=["Coverage", "Confidence Level"])
        plt.savefig("plots/frechet/coverage_{}_{}_{}.png".format(\
            frec_params[i], alph, Fu),\
            bbox_inches="tight")
        plt.clf()
