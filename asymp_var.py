import numpy as np
from frechet import Frechet
from confidence_intervals import V
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties

if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    np.set_printoptions(linewidth=1000)
    np.set_printoptions(threshold=10000)
    alph = [0.99, 0.999]
    dists = [Frechet(2.25), Frechet(2.5), Frechet(3), Frechet(4)]
    n = np.arange(10000, 100001, 1000)
    k = np.ceil(n**(2/3)).astype(int)

    plt.style.use('seaborn')
    plt.rc('axes', titlesize=8)     # fontsize of the axes title
    plt.rc('axes', labelsize=6)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=6)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=6)    # fontsize of the tick labels
    plt.rc('legend', fontsize=6)    # fontsize of the tick labels
    plt.rc('font', family='serif')

    # uncomment this line for Latex rendering
    #plt.rc('text', usetex=True)

    n_rows = len(alph)
    n_cols = len(dists)
    fig, axs = plt.subplots(n_rows, n_cols, sharex=True, figsize=(7, 3))

    for i in range(n_rows):
        al = alph[i]
        axs[i,0].set_ylabel('asymptotic variance')
        for j in range(n_cols):
            D = dists[j]
            axs[i,j].set_title('gamma={}, alpha={}'.format(D.gamma, al))
            beta = k/n/(1-al)
            avar_sa = D.avar(al)/n
            avar_ev = D.a(n/k)**2 * np.array([V(D.xi, b) for b in beta])/k
            axs[i,j].plot(n, avar_sa)
            axs[i,j].plot(n, avar_ev)
            axs[i,j].legend(['SA', 'POT'])

    for i in range(n_cols):
        axs[n_rows-1, i].set_xlabel('sample size')

    plt.tight_layout(pad=0.5)
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))

    fig.savefig('plots/asymp_var.pdf', format='pdf', bbox_inches='tight')

    plt.show()
    plt.clf()
