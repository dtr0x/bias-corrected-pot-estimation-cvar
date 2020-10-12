from param_search_util import *

if __name__ == '__main__':
    s = 100
    n = 25000

    # Frechet distributions
    p = np.linspace(1.25, 3, 8)

    T = []
    for D in dists:
        np.random.seed(0)
        data = D.rand((s, n))
        r = D.rho
        r_est, r_mse, k_rho = rho_search(data, r)
        print("Frechet({:.2f}): ".format(D.gamma))
        print(r_est, r_mse, k_rho)
        print('')
        T.append(theta(n, k_rho))
