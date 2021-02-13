import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties
from frechet import Frechet
from burr import Burr
from half_t import HalfT
from confidence_intervals import V, conf_int


'''Compute and plot coverage probabilities for confidence intervals.'''


def cov_prob(cvar, intervals):
    return np.where((cvar >= intervals[:,0]) & \
        (cvar <= intervals[:,1]))[0].size/intervals.shape[0]


if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    np.set_printoptions(linewidth=1000)
    np.set_printoptions(threshold=10000)
    # CVaR estimates
    cvars_est = np.load('data/cvars.npy')

    # CVaR level
    alph = 0.998

    xi = 2/3
    d = np.array([4, 3, 2.25, 0.75, 0.45])
    c = 1/(xi*d)
    burr_params = np.around([(i,j) for i,j in zip(c,d)], 2)
    fs_parms = np.linspace(1.5, 2.5, 5)
    dists = [Burr(*p) for p in burr_params]
    dists += [Frechet(p) for p in fs_parms]
    dists += [HalfT(p) for p in fs_parms]

    dist_titles = [d.get_label() for d in dists]

    dist_cvars = [d.cvar(alph) for d in dists]

    # sample sizes to test CVaR estimation
    sampsizes = np.linspace(5000, 50000, 10).astype(int)

    # unbiased POT estimator
    upot = cvars_est[2]
    # bias-corrected xi estimates
    xi = cvars_est[3]
    # bias-corrected sigma estimates
    sigma = cvars_est[4]
    # number of threshold excesses used
    n_excesses = cvars_est[6]
    # beta values
    beta = n_excesses/sampsizes/(1-alph)

    # compute confidence intervals
    cp = []
    for i in range(upot.shape[0]):
        cvar_true = dist_cvars[i]
        cp_dist = []
        for j in range(upot.shape[2]):
            c = upot[i,:,j]
            idx = np.where(~np.isnan(c))[0]
            c = c[idx]
            x = xi[i,idx,j]
            s = sigma[i,idx,j]
            k = n_excesses[i,idx,j]
            b = beta[i,idx,j]
            v = np.asarray([V(y,z) for y,z in zip(x,b)])
            intervals = np.asarray(conf_int(c,s,v,k)).transpose()
            cp_dist.append(cov_prob(cvar_true, intervals))
        cp.append(cp_dist)

    cp = np.asarray(cp)

    plt.style.use('seaborn')
    plt.rc('axes', titlesize=8)     # fontsize of the axes title
    plt.rc('axes', labelsize=6)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=4)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=4)    # fontsize of the tick labels
    plt.rc('legend', fontsize=4)    # fontsize of the tick labels
    plt.rc('font', family='serif')

    # uncomment this line for Latex rendering
    #plt.rc('text', usetex=True)

    n_rows = 3
    n_cols = 5
    fig, axs = plt.subplots(n_rows, n_cols, sharex=True, figsize=(9, 4))

    for i in range(n_rows):
        axs[i,0].set_ylabel('coverage probability')
        for j in range(n_cols):
            idx = i*n_cols+j
            axs[i,j].plot(sampsizes, cp[idx], linestyle='--', linewidth=0.5, marker='.', markersize=4, color='k')
            axs[i,j].plot(sampsizes, [0.95]*len(sampsizes), linewidth=1, color='k')
            axs[i,j].set_title(dist_titles[idx])
            axs[i,j].set_ylim(min(min(cp[idx]), 0.8))
            axs[i,j].legend(['coverage', 'confidence level'])


    for i in range(n_cols):
        axs[n_rows-1, i].set_xlabel('sample size')

    plt.tight_layout(pad=0.5)
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    fig.savefig('plots/cov_prob.pdf', format='pdf', bbox_inches='tight')

    plt.show()
    plt.clf()
