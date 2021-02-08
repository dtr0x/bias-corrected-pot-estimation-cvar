import numpy as np
from estimators import rho_est

# Compute the length of the longest subsequence of x of same values and the corresponding indices
def max_run_len(x):
    diffs = np.diff(x)

    # get indices of consecutive value run ends (non-inclusive)
    run_ends = np.where(diffs)[0] + 1
    run_intervals = np.hstack((0, run_ends, x.size))

    interval_idx = np.diff(run_intervals).argmax()
    k_min = run_intervals[interval_idx]
    k_max = run_intervals[interval_idx+1]
    max_len = np.diff(run_intervals).max()

    return k_min, k_max, max_len

# def ada_rho(x, min_m=100, step=50):
#     n = x.size
#     num = int((n - min_m)/step)
#     sample_fracs = np.linspace(min_m, n-1, num).astype(int)

# Adaptive estimation of the second-order parameter based on stable sample paths
def ada_rho(x):
    n = x.size
    sample_fracs = np.linspace(0.1*n, n-1, 100).astype(int)
    tau = np.linspace(-1.5, 1.5, 13)

    rho_hat = []
    for t in tau:
        rho_hat_t = []
        for k in sample_fracs:
            rho_hat_t.append(rho_est(x, k, t))
        rho_hat.append(rho_hat_t)

    rho_hat = np.asarray(rho_hat)
    rho_round = np.around(rho_hat, 1)

    sample_path_data = np.apply_along_axis(max_run_len, 1, rho_round)
    max_len_idx = sample_path_data[:,-1].argmax()
    j1, j2 = sample_path_data[max_len_idx, :-1]
    chosen_rho_path = rho_hat[max_len_idx, j1:j2]

    rho_final = np.median(chosen_rho_path)

    return rho_final
