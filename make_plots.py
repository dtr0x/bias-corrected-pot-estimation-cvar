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
    rhos = np.linspace(-0.25, -2, 8)
    d = -1/rhos
    c = 1/(xi*d)
    params = [(i,j) for i,j in zip(c,d)]
    burr_dists = [Burr(*p) for p in params]

    # Frechet distributions
    gamma = np.linspace(1.25, 3, 8)
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

    plt.style.use('seaborn')
    plt.rc('axes', titlesize=10)     # fontsize of the axes title
    plt.rc('axes', labelsize=8)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=6)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=6)    # fontsize of the tick labels
    plt.rc('font', family='san-serif')

    # uncomment this line for Latex rendering
    #plt.rc('text', usetex=True)

    n_dists = len(burr_dists)

    fig, axs = plt.subplots(2, n_dists, sharex=True)

    # RMSE plots
    for i in range(len(burr_dists)):
        # Burr plots
        axs[0,i].plot(N, burr_rmse[i,0], linestyle='--', linewidth=1, marker='.', color='k')
        axs[0,i].plot(N, burr_rmse[i,1], linestyle='--', linewidth=1, marker='.', color='r')
        axs[0,i].plot(N, burr_rmse[i,2], linestyle='--', linewidth=1, marker='.', color='b')
        # Frechet plots
        axs[1,i].plot(N, frec_rmse[i,0], linestyle='--', linewidth=1, marker='.', color='k')
        axs[1,i].plot(N, frec_rmse[i,1], linestyle='--', linewidth=1, marker='.', color='r')
        axs[1,i].plot(N, frec_rmse[i,2], linestyle='--', linewidth=1, marker='.', color='b')
        axs[1,i].set_xlabel('sample size')

    axs[0,0].set_ylabel('RMSE')
    axs[1,0].set_ylabel('RMSE')
    axs[0,0].legend(['UPOT', 'BPOT', 'SA'])

    plt.tight_layout()
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    fig.savefig('plots/rmse.pdf', format='pdf', bbox_inches='tight')

    plt.show()
    plt.clf()

    # bias Burr plots
    # fig, axs = plt.subplots(2, 2, sharex=True)
    #
    # axs[0,0].set_title(burr_labels[0])
    # axs[0,1].set_title(burr_labels[1])
    # axs[1,0].set_title(burr_labels[2])
    # axs[1,1].set_title(burr_labels[3])
    #
    # axs[0,0].plot(N, burr_bias[0,0], linestyle='--', linewidth=1, marker='.', color='k')
    # axs[0,0].plot(N, burr_bias[0,1], linestyle='--', linewidth=1, marker='.', color='r')
    # axs[0,0].plot(N, burr_bias[0,2], linestyle='--', linewidth=1, marker='.', color='b')
    #
    # axs[0,1].plot(N, burr_bias[1,0], linestyle='--', linewidth=1, marker='.', color='k')
    # axs[0,1].plot(N, burr_bias[1,1], linestyle='--', linewidth=1, marker='.', color='r')
    # axs[0,1].plot(N, burr_bias[1,2], linestyle='--', linewidth=1, marker='.', color='b')
    #
    # axs[1,0].plot(N, burr_bias[2,0], linestyle='--', linewidth=1, marker='.', color='k')
    # axs[1,0].plot(N, burr_bias[2,1], linestyle='--', linewidth=1, marker='.', color='r')
    # axs[1,0].plot(N, burr_bias[2,2], linestyle='--', linewidth=1, marker='.', color='b')
    #
    # axs[1,1].plot(N, burr_bias[3,0], linestyle='--', linewidth=1, marker='.', color='k')
    # axs[1,1].plot(N, burr_bias[3,1], linestyle='--', linewidth=1, marker='.', color='r')
    # axs[1,1].plot(N, burr_bias[3,2], linestyle='--', linewidth=1, marker='.', color='b')
    #
    # axs[1,0].set_xlabel('sample size')
    # axs[1,1].set_xlabel('sample size')
    # axs[0,0].set_ylabel('bias')
    # axs[1,0].set_ylabel('bias')
    # axs[0,0].legend(['UPOT', 'POT', 'SA'])
    #
    # plt.tight_layout()
    # plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    # fig.savefig('plots/burr_bias.pdf', format='pdf', bbox_inches='tight')
    #
    # fig, axs = plt.subplots(2, 2, sharex=True)
    #
    # axs[0,0].set_title(frec_labels[0])
    # axs[0,1].set_title(frec_labels[1])
    # axs[1,0].set_title(frec_labels[2])
    # axs[1,1].set_title(frec_labels[3])
    #
    # axs[0,0].plot(N, frec_rmse[0,0], linestyle='--', linewidth=1, marker='.', color='k')
    # axs[0,0].plot(N, frec_rmse[0,1], linestyle='--', linewidth=1, marker='.', color='r')
    # axs[0,0].plot(N, frec_rmse[0,2], linestyle='--', linewidth=1, marker='.', color='b')
    #
    # axs[0,1].plot(N, frec_rmse[1,0], linestyle='--', linewidth=1, marker='.', color='k')
    # axs[0,1].plot(N, frec_rmse[1,1], linestyle='--', linewidth=1, marker='.', color='r')
    # axs[0,1].plot(N, frec_rmse[1,2], linestyle='--', linewidth=1, marker='.', color='b')
    #
    # axs[1,0].plot(N, frec_rmse[2,0], linestyle='--', linewidth=1, marker='.', color='k')
    # axs[1,0].plot(N, frec_rmse[2,1], linestyle='--', linewidth=1, marker='.', color='r')
    # axs[1,0].plot(N, frec_rmse[2,2], linestyle='--', linewidth=1, marker='.', color='b')
    #
    # axs[1,1].plot(N, frec_rmse[3,0], linestyle='--', linewidth=1, marker='.', color='k')
    # axs[1,1].plot(N, frec_rmse[3,1], linestyle='--', linewidth=1, marker='.', color='r')
    # axs[1,1].plot(N, frec_rmse[3,2], linestyle='--', linewidth=1, marker='.', color='b')
    #
    # axs[1,0].set_xlabel('sample size')
    # axs[1,1].set_xlabel('sample size')
    # axs[0,0].set_ylabel('RMSE')
    # axs[1,0].set_ylabel('RMSE')
    # axs[0,0].legend(['UPOT', 'POT', 'SA'])
    #
    # plt.tight_layout()
    # plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    # fig.savefig('plots/frec_rmse.pdf', format='pdf', bbox_inches='tight')
    #
    # # bias frec plots
    # fig, axs = plt.subplots(2, 2, sharex=True)
    #
    # axs[0,0].set_title(frec_labels[0])
    # axs[0,1].set_title(frec_labels[1])
    # axs[1,0].set_title(frec_labels[2])
    # axs[1,1].set_title(frec_labels[3])
    #
    # axs[0,0].plot(N, frec_bias[0,0], linestyle='--', linewidth=1, marker='.', color='k')
    # axs[0,0].plot(N, frec_bias[0,1], linestyle='--', linewidth=1, marker='.', color='r')
    # axs[0,0].plot(N, frec_bias[0,2], linestyle='--', linewidth=1, marker='.', color='b')
    #
    # axs[0,1].plot(N, frec_bias[1,0], linestyle='--', linewidth=1, marker='.', color='k')
    # axs[0,1].plot(N, frec_bias[1,1], linestyle='--', linewidth=1, marker='.', color='r')
    # axs[0,1].plot(N, frec_bias[1,2], linestyle='--', linewidth=1, marker='.', color='b')
    #
    # axs[1,0].plot(N, frec_bias[2,0], linestyle='--', linewidth=1, marker='.', color='k')
    # axs[1,0].plot(N, frec_bias[2,1], linestyle='--', linewidth=1, marker='.', color='r')
    # axs[1,0].plot(N, frec_bias[2,2], linestyle='--', linewidth=1, marker='.', color='b')
    #
    # axs[1,1].plot(N, frec_bias[3,0], linestyle='--', linewidth=1, marker='.', color='k')
    # axs[1,1].plot(N, frec_bias[3,1], linestyle='--', linewidth=1, marker='.', color='r')
    # axs[1,1].plot(N, frec_bias[3,2], linestyle='--', linewidth=1, marker='.', color='b')
    #
    # axs[1,0].set_xlabel('sample size')
    # axs[1,1].set_xlabel('sample size')
    # axs[0,0].set_ylabel('bias')
    # axs[1,0].set_ylabel('bias')
    # axs[0,0].legend(['UPOT', 'POT', 'SA'])
    #
    # plt.tight_layout()
    # plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    # fig.savefig('plots/frec_bias.pdf', format='pdf', bbox_inches='tight')
