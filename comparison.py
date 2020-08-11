import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from burr import Burr
from frechet import Frechet
from scipy.stats import norm
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties

if __name__ == '__main__':
    dists = [Burr(0.5, 3), Burr(0.75, 2), Frechet(1.2), Frechet(1.6)]
    labels = ['Burr(0.5, 3)', 'Burr(0.75, 2)', 'Fr\\\'echet(1.2)', 'Fr\\\'echet(1.6)']
    n_samples = 5000
    sampsizes = np.array([1000, 2000, 5000, 10000, 25000, 50000, 100000])
    alph = 0.999
    delt = 0.1
    cvar_true = [d.cvar(alph) for d in dists]

    sa = np.load('data/cvar_sa.npz')['cvars']
    sa_means = np.nanmean(sa, axis=1)
    sa_rmse = []
    for i in range(len(cvar_true)):
        mse = np.nanmean((sa[i]-cvar_true[i])**2, axis=0)
        sa_rmse.append(np.sqrt(mse))
    sa_rmse = np.asarray(sa_rmse)

    calc = np.load('data/calc.npz')
    xi_est = calc['xi_est']
    sig_est = calc['sig_est']
    cvar_pot_est = calc['cvar_pot_est']
    cvar_pot_est[xi_est >= 0.9] = np.nan
    approx_error = calc['approx_error']
    avar = calc['avar']
    cvar_unbiased = cvar_pot_est - approx_error
    asymp_stderr = sig_est * np.sqrt(avar)
    ci_lower = cvar_unbiased - asymp_stderr*norm.ppf(1-delt/2)
    ci_upper = cvar_unbiased + asymp_stderr*norm.ppf(1-delt/2)

    ev_rmse = []
    for i in range(len(cvar_true)):
        mse = np.nanmean((cvar_unbiased[i]-cvar_true[i])**2, axis=0)
        ev_rmse.append(np.sqrt(mse))
    ev_rmse = np.asarray(ev_rmse)

    cvar_means = np.nanmean(cvar_unbiased, axis=1)
    ci_l_means = np.nanmean(ci_lower, axis=1)
    ci_u_means = np.nanmean(ci_upper, axis=1)

    ci_l_emp = np.nanquantile(cvar_unbiased, delt/2, axis=1)
    ci_u_emp = np.nanquantile(cvar_unbiased, 1-delt/2, axis=1)

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
    fig, axs = plt.subplots(2, n_dists, sharex=True, figsize=(6.5, 3))

    for i in range(n_dists):
        axs[0,i].set_title(labels[i])
        axs[0,i].plot(sampsizes, [cvar_true[i]]*len(sampsizes), linestyle='-', linewidth=1, color='k')
        axs[0,i].plot(sampsizes, cvar_means[i], linestyle='--', linewidth=1, color='k')
        axs[0,i].plot(sampsizes, ci_l_means[i], linestyle='--', linewidth=1, color='r')
        axs[0,i].plot(sampsizes, ci_u_means[i], linestyle='--', linewidth=1, color='r')
        axs[0,i].plot(sampsizes, ci_l_emp[i], linestyle='--', linewidth=1, color='b')
        axs[0,i].plot(sampsizes, ci_u_emp[i], linestyle='--', linewidth=1, color='b')
        axs[0,i].set_ylabel('CVaR')

        axs[1,i].plot(sampsizes, ev_rmse[i], linestyle='--', linewidth=1, color='green')
        axs[1,i].plot(sampsizes, sa_rmse[i], linestyle='--', linewidth=1, color='orange')
        axs[1,i].set_ylabel('RMSE')

    for ax in axs.flat:
        ax.set(xlabel='Sample size')
        ax.label_outer()

    custom_lines = [
        Line2D([0], [0], linestyle='-', linewidth=1, color='k'),
        Line2D([0], [0], linestyle='--', linewidth=1, color='k'),
        Line2D([0], [0], linestyle='--', linewidth=1, color='r'),
        Line2D([0], [0], linestyle='--', linewidth=1, color='b'),
        Line2D([0], [0], linestyle='--', linewidth=1, color='green'),
        Line2D([0], [0], linestyle='--', linewidth=1, color='orange')
        ]
    line_labels = [
        "CVaR_0.999", "POT estimate", "Estimated CI", "Empirical CI", "POT RMSE", "SA RMSE"]
    fontP = FontProperties()
    fontP.set_size('small')
    # Create the legend
    plt.subplots_adjust(right=0.85)
    fig.legend(custom_lines,  line_labels, borderaxespad=0, prop=fontP, loc='right')#bbox_to_anchor=(1, 0.75))
    #plt.tight_layout()
    fig.savefig('output/comparison.pdf', format='pdf', bbox_inches='tight')
    plt.close("all")
