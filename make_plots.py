import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties
from frechet import Frechet
from burr import Burr

def rmse(x, true):
    return np.sqrt(np.nanmean((x-true)**2, axis=0))

def bias(x, true):
    return np.abs(np.nanmean(x, axis=0) - true)

def nan_rate(x):
    return np.isnan(x).sum(axis=0)/x.shape[0]

def cvar_rmse(cvars, cvars_true):
    r = []
    for i in range(len(cvars_true)):
        r1 = rmse(cvars[0,i], cvars_true[i])
        r2 = rmse(cvars[1,i], cvars_true[i])
        r3 = rmse(cvars[2,i], cvars_true[i])
        r.append([r1,r2,r3])
    return np.array(r)

def cvar_bias(cvars, cvars_true):
    b = []
    for i in range(len(cvars_true)):
        b1 = bias(cvars[0,i], cvars_true[i])
        b2 = bias(cvars[1,i], cvars_true[i])
        b3 = bias(cvars[2,i], cvars_true[i])
        b.append([b1,b2,b3])
    return np.array(b)

def cvar_nans(cvars):
    nas = []
    for i in range(cvars.shape[1]):
        n1 = nan_rate(cvars[0,i])
        n2 = nan_rate(cvars[1,i])
        nas.append([n1,n2])
    return np.array(nas)

if __name__ == '__main__':
    burr_cvars = np.load('data/burr_cvars.npy')
    frec_cvars = np.load('data/frec_cvars.npy')

    # Burr distributions
    xi = 2/3
    rho = -np.array([0.25, 0.4, 0.75, 1.5, 2])
    d = -1/rho
    c = 1/(xi*d)
    params = [(i,j) for i,j in zip(c,d)]
    burr_dists = [Burr(*p) for p in params]

    # Frechet distributions
    gamma = np.linspace(1.5, 2.5, 5)
    frec_dists = [Frechet(p) for p in gamma]

    # sample sizes to test CVaR estimation
    N = np.array([10000, 20000, 30000, 40000, 50000])

    # CVaR level
    alph = 0.999
    burr_cvars_true = [d.cvar(alph) for d in burr_dists]
    frec_cvars_true = [d.cvar(alph) for d in frec_dists]

    # rmse
    burr_rmse = cvar_rmse(burr_cvars, burr_cvars_true)
    frec_rmse = cvar_rmse(frec_cvars, frec_cvars_true)
    # bias
    burr_bias = cvar_bias(burr_cvars, burr_cvars_true)
    frec_bias = cvar_bias(frec_cvars, frec_cvars_true)
    # nan percentage
    burr_nanrate = cvar_nans(burr_cvars)
    frec_nanrate = cvar_nans(frec_cvars)

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
    # n_dists = len(burr_dists)
    # fig, axs = plt.subplots(4, n_dists, sharex=True, figsize=(6.8, 4.2))
    #
    # burr_titles = ["Burr({}, {})".format(c,d) for c,d in np.around(params, 2)]
    # frec_titles = ["Frechet({})".format(g) for g in np.around(gamma, 2)]
    #
    # for i in range(len(burr_dists)):
    #     # Burr plots
    #     # RMSE
    #     axs[0,i].plot(N, burr_rmse[i,0], linestyle='--', linewidth=0.5, marker='.', markersize=5, color='k')
    #     axs[0,i].plot(N, burr_rmse[i,1], linestyle='--', linewidth=0.5, marker='.', markersize=5, color='r')
    #     axs[0,i].plot(N, burr_rmse[i,2], linestyle='--', linewidth=0.5, marker='.', markersize=5, color='b')
    #     # Bias
    #     axs[1,i].plot(N, burr_bias[i,0], linestyle='--', linewidth=0.5, marker='.', markersize=5, color='k')
    #     axs[1,i].plot(N, burr_bias[i,1], linestyle='--', linewidth=0.5, marker='.', markersize=5, color='r')
    #     axs[1,i].plot(N, burr_bias[i,2], linestyle='--', linewidth=0.5, marker='.', markersize=5, color='b')
    #
    #     # Frechet plots
    #     # RMSE
    #     axs[2,i].plot(N, frec_rmse[i,0], linestyle='--', linewidth=0.5, marker='.', markersize=5, color='k')
    #     axs[2,i].plot(N, frec_rmse[i,1], linestyle='--', linewidth=0.5, marker='.', markersize=5, color='r')
    #     axs[2,i].plot(N, frec_rmse[i,2], linestyle='--', linewidth=0.5, marker='.', markersize=5, color='b')
    #     # Bias
    #     axs[3,i].plot(N, frec_bias[i,0], linestyle='--', linewidth=0.5, marker='.', markersize=5, color='k')
    #     axs[3,i].plot(N, frec_bias[i,1], linestyle='--', linewidth=0.5, marker='.', markersize=5, color='r')
    #     axs[3,i].plot(N, frec_bias[i,2], linestyle='--', linewidth=0.5, marker='.', markersize=5, color='b')
    #
    #     axs[3,i].set_xlabel('sample size')
    #
    #     axs[0,i].set_title(burr_titles[i])
    #     axs[2,i].set_title(frec_titles[i])

    # axs[0,0].set_ylabel('RMSE')
    # axs[1,0].set_ylabel('absolute bias')
    # axs[2,0].set_ylabel('RMSE')
    # axs[3,0].set_ylabel('absolute bias')
    # axs[0,0].legend(['UPOT', 'BPOT', 'SA'])
    #
    # plt.tight_layout(pad=0.5)
    # plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    # fig.savefig('plots/all.pdf', format='pdf', bbox_inches='tight')
    #
    # plt.show()
    # plt.clf()
