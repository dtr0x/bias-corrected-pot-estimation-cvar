import numpy as np
import matplotlib.pyplot as plt
from gen_data import gen_data
from burr import Burr
from frechet import Frechet
from confidence_intervals import confidence_intervals

if __name__ == '__main__':
    alph = 0.999
    delt = 0.1
    burr_params = ((3/4, 2), (1.5, 1))
    frec_params = (1.25, 2)
    n_samples = 5000
    sampsizes = np.array([1000, 2000, 5000, 10000, 25000, 50000, 100000])
    n_excesses = np.ceil(sampsizes**(2/3)).astype(int)
    Fus = 1 - n_excesses/sampsizes
    burr_dists = [Burr(*p) for p in burr_params]
    frec_dists = [Frechet(p) for p in frec_params]
    burr_cvars = [b.cvar(alph) for b in burr_dists]
    frec_cvars = [f.cvar(alph) for f in frec_dists]

    burr_data, frec_data, burr_cvars_evt, frec_cvars_evt = \
    gen_data(burr_dists, frec_dists, alph, n_samples, sampsizes, n_excesses)

    burr_ci, burr_cp = confidence_intervals(burr_dists, n_excesses, Fus, \
                        alph, delt, burr_cvars_evt)
    frec_ci, frec_cp = confidence_intervals(frec_dists, n_excesses, Fus, \
                        alph, delt, frec_cvars_evt)

    burr_cvar_means = np.nanmean(burr_cvars_evt, axis=1)
    for i in range(burr_cvar_means.shape[0]):
        p1, = plt.plot(n_excesses, burr_cvar_means[i], \
                linestyle='-', marker='.', color='k')
        p2, = plt.plot(n_excesses, burr_ci[i, :, 0], \
                linestyle='--', marker='.', fillstyle='none', color='#383838')
        p3, = plt.plot(n_excesses, burr_ci[i, :, 1], \
                linestyle='--', marker='.', fillstyle='none', color='#383838')
        p4, = plt.plot(n_excesses, [burr_cvars[i]]*len(n_excesses), \
                linestyle='-', color='#00208a')
        plt.xlabel("number of excesses")
        plt.ylabel("CVaR")
        plt.title("Burr{}, alpha={}".format(burr_params[i], alph))
        plt.legend([p1, p2, p4], \
            ["Mean CVaR", "Mean Confidence Intervals", "True CVaR"])
        plt.savefig("plots/burr/intervals_{}_{}_{}.png".format(\
            burr_params[i][0], burr_params[i][1], alph),\
            bbox_inches="tight")
        plt.clf()
        plt.plot(n_excesses, burr_cp[i], \
                linestyle='--', color='k', marker='.')
        plt.plot(n_excesses, [1-delt]*len(n_excesses), linestyle='-', color='k')
        plt.xlabel("number of excesses")
        plt.ylabel("coverage probability")
        plt.title("Burr{}, alpha={}".format(burr_params[i], alph))
        plt.legend(labels=["Coverage", "Confidence Level"])
        plt.savefig("plots/burr/coverage_{}_{}_{}.png".format(\
            burr_params[i][0], burr_params[i][1], alph),\
            bbox_inches="tight")
        plt.clf()

    frec_cvar_means = np.nanmean(frec_cvars_evt, axis=1)
    for i in range(frec_cvar_means.shape[0]):
        p1, = plt.plot(n_excesses, frec_cvar_means[i], \
                linestyle='-', marker='.', color='k')
        p2, = plt.plot(n_excesses, frec_ci[i, :, 0], \
                linestyle='--', marker='.', fillstyle='none', color='#383838')
        p3, = plt.plot(n_excesses, frec_ci[i, :, 1], \
                linestyle='--', marker='.', fillstyle='none', color='#383838')
        p4, = plt.plot(n_excesses, [frec_cvars[i]]*len(n_excesses), \
                linestyle='-', color='#00208a')
        plt.xlabel("number of excesses")
        plt.ylabel("CVaR")
        plt.title("Frechet({}), alpha={}".format(frec_params[i], alph))
        plt.legend([p1, p2, p4], \
            ["Mean CVaR", "Mean Confidence Intervals", "True CVaR"])
        plt.savefig("plots/frechet/intervals_{}_{}.png".format(\
            frec_params[i], alph),\
            bbox_inches="tight")
        plt.clf()
        plt.plot(n_excesses, frec_cp[i], \
                linestyle='--', color='k', marker='.')
        plt.plot(n_excesses, [1-delt]*len(n_excesses), linestyle='-', color='k')
        plt.xlabel("number of excesses")
        plt.ylabel("coverage probability")
        plt.title("Frechet({}), alpha={}".format(frec_params[i], alph))
        plt.legend(labels=["Coverage", "Confidence Level"])
        plt.savefig("plots/frechet/coverage_{}_{}.png".format(\
            frec_params[i], alph),\
            bbox_inches="tight")
        plt.clf()
