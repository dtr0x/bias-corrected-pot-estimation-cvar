import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

if __name__ == '__main__':
    data = np.load('data/results.npz', allow_pickle=True)
    cov_prob = data['cov_prob']
    cvar_est = data['cvar_est']
    approx_error = data['approx_error']
    avar = data['avar']
    dists = data['dists']
    labels = data['dist_labels']
    cvar_true = data['cvar_true']
    nan_rate = data['nan_rate']
    k = data['n_excesses']
    delt = data['delt']
    alph = data['alph']
    rmse = data['rmse']

    plt.style.use('seaborn')
    plt.rc('axes', titlesize=10)     # fontsize of the axes title
    plt.rc('axes', labelsize=8)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=6)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=6)    # fontsize of the tick labels
    plt.rc('font', family='Times New Roman')

    # uncomment this line for Latex rendering
    #plt.rc('text', usetex=True)

    n_dists = len(labels)
    # output dims for paper
    fig, axs = plt.subplots(1, n_dists, figsize=(6.2, 1.5))

    for i in range(n_dists):
        axs[i].plot(k, cov_prob[i], linestyle='--', color='k', marker='.', \
                    markersize=6, linewidth=1)
        axs[i].plot(k, [1-delt]*len(k), linestyle='-', color='k', \
                    linewidth=0.5)
        axs[i].set_title(labels[i])

    for ax in axs.flat:
        ax.set(xlabel='k', ylabel='coverage probability')

    for ax in axs.flat:
        ax.label_outer()

    #fig.savefig('output/cov_prob.pdf', format='pdf', bbox_inches='tight')
    plt.close("all")

    with open('output/latex_output.txt', 'w') as f:
        for d, c, ae, av, cp, l, ct in \
            zip(dists, cvar_est, approx_error, avar, cov_prob, labels, \
                cvar_true):
                f.write('\\begin{table}[H]\n')
                f.write('\\caption{{Values for the {:s} distribution, which has $\\xi={:.3f}$, $\\rho={:.3f}$, $c_\\alpha={:.2f}$.}}\n'.format(l, d.xi, d.rho, ct))
                f.write('\\centering\n')
                f.write('\\begin{tabular}{cccccc} \\\\ \\toprule\n')
                f.write('$k$ & $\\hat{c}^{(n)}_\\alpha - \\hat{\\epsilon}_{u,\\alpha}$ & $\\hat{\\epsilon}_{u,\\alpha}$ & $\\hat{V}/k$ & $\\hat{P}_{\\delta}^n(N)$ \\\\ \\midrule\n')
                for i in range(2, len(k)):
                    f.write('{:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.3f} \\\\\n'.\
                            format(k[i], c[i], ae[i], av[i], cp[i]))
                f.write('\\bottomrule \\end{tabular} \\end{table}\n\n')
