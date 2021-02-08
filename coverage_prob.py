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
    # CVaR estimates (fill NaNs)
    cvars_est = np.load('data/cvars.npy')
    cvars_est[1][np.where(np.isnan(cvars_est[1]))] = cvars_est[0][np.where(np.isnan(cvars_est[1]))]
    cvars_est[2][np.where(np.isnan(cvars_est[2]))] = cvars_est[1][np.where(np.isnan(cvars_est[2]))]

    # CVaR level
    alph = 0.999

    c = np.array([0.45, 0.6, 0.7, 1.45, 2.75])
    d = np.array([3, 2.5, 2, 1, 0.5])
    dists = [Burr(i,j) for (i,j) in zip(c,d)]
    dists += [Frechet(1.25), Frechet(1.5), Frechet(1.75), Frechet(2), Frechet(2.25), \
             HalfT(1.25), HalfT(1.5), HalfT(1.75), HalfT(2), HalfT(2.25)]

    dist_titles = [d.get_label() for d in dists]

    dist_cvars = [d.cvar(alph) for d in dists]

    # sample sizes to test CVaR estimation
    sampsizes = np.linspace(10000, 50000, 5).astype(int)

    # unbiased POT estimator
    upot = cvars_est[2]
    # bias-corrected Xi estimates
    xi = cvars_est[3]
    # bias-corrected Sigma estimates
    sigma = cvars_est[4]

    theta = 0.8
    k = np.ceil(sampsizes**theta).astype(int)
    tp = 1-k/sampsizes
    beta = (1-tp)/(1-alph)

    # coverage probabilities for each distribution
    cp = []

    # compute confidence intervals
    for d in range(len(dists)):
        intervals = []
        for i in range(xi.shape[1]):
            V_est = [V(x,b) for x,b in zip(xi[d,i],beta)]
            ci = [conf_int(c, s, v, j) for c,s,v,j in zip(upot[d,i], sigma[d,i], V_est, k)]
            intervals.append(ci)
        intervals = np.asarray(intervals)

        # compute coverage probability
        c = dist_cvars[d]
        cp.append([cov_prob(c, intervals[:,i,:]) for i in range(intervals.shape[1])])

    cp = np.asarray(cp)



    # dist_rmse = cvar_rmse(cvars_est, dist_cvars)
    # dist_bias = cvar_bias(cvars_est, dist_cvars)
    #
    # plt.style.use('seaborn')
    # plt.rc('axes', titlesize=8)     # fontsize of the axes title
    # plt.rc('axes', labelsize=6)    # fontsize of the x and y labels
    # plt.rc('xtick', labelsize=4)    # fontsize of the tick labels
    # plt.rc('ytick', labelsize=4)    # fontsize of the tick labels
    # plt.rc('legend', fontsize=4)    # fontsize of the tick labels
    # plt.rc('font', family='serif')
    #
    # # uncomment this line for Latex rendering
    # #plt.rc('text', usetex=True)
    #
    # n_rows = 6
    # n_cols = 5
    # fig, axs = plt.subplots(n_rows, n_cols, sharex=True, figsize=(9, 8))
    #
    # for i in np.arange(0, n_rows, 2):
    #     axs[i,0].set_ylabel('RMSE')
    #     axs[i+1,0].set_ylabel('absolute bias')
    #     for j in range(n_cols):
    #         idx = int(i/2)*n_cols+j
    #         axs[i,j].plot(sampsizes, dist_rmse[idx,0], linestyle='--', linewidth=0.5, marker='.', markersize=5, color='b')
    #         axs[i,j].plot(sampsizes, dist_rmse[idx,1], linestyle='--', linewidth=0.5, marker='.', markersize=5, color='r')
    #         axs[i,j].plot(sampsizes, dist_rmse[idx,2], linestyle='--', linewidth=0.5, marker='.', markersize=5, color='k')
    #         axs[i+1,j].plot(sampsizes, dist_bias[idx,0], linestyle='--', linewidth=0.5, marker='.', markersize=5, color='b')
    #         axs[i+1,j].plot(sampsizes, dist_bias[idx,1], linestyle='--', linewidth=0.5, marker='.', markersize=5, color='r')
    #         axs[i+1,j].plot(sampsizes, dist_bias[idx,2], linestyle='--', linewidth=0.5, marker='.', markersize=5, color='k')
    #         axs[i,j].set_title(dist_titles[idx])
    #
    #     axs[i,0].legend(['SA', 'BPOT', 'UPOT'])
    #
    #
    # for i in range(n_cols):
    #     axs[n_rows-1, i].set_xlabel('sample size')
    #
    # plt.tight_layout(pad=0.5)
    # plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    # fig.savefig('plots/dist_plots.pdf', format='pdf', bbox_inches='tight')
    #
    # plt.show()
    # plt.clf()
